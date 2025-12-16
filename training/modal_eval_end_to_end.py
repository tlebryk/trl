"""
End-to-end evaluation on Modal: inference + evaluation.

This script runs both:
1. Inference (GPU) - generates completions
2. Evaluation (CPU) - compiles/runs and computes metrics

Convenience wrapper around modal_inference.py and eval_completions.py.
"""
import modal
import json
import os
from typing import Optional

# Inference image (GPU, no compiler)
inference_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch",
        "transformers",
        "peft",
        "datasets",
        "accelerate",
    )
)

# Evaluation image (CPU, with compiler)
eval_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("g++", "build-essential")
    .pip_install("scipy")  # Required by cpp_utils
    .add_local_dir("cpp_pipeline", "/root/cpp_pipeline")
)

# Volume for data storage
volume = modal.Volume.from_name("dpo-training-vol", create_if_missing=True)

app = modal.App("dpo-eval-e2e", image=inference_image)

EXPERIMENTS_DIR = "/data/experiments"
INFERENCE_OUTPUT_DIR = "/data/inference_results"


@app.function(
    gpu="T4",
    timeout=7200,
    volumes={"/data": volume},
    image=inference_image
)
def generate_completions(
    adapter_path: Optional[str],
    num_problems: Optional[int],
    num_samples: int,
    temperature: float,
    run_name: Optional[str]
):
    """Generate completions (same as modal_inference.py)."""
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from datetime import datetime

    print("="*80)
    print("STEP 1: INFERENCE")
    print("="*80)
    print(f"Adapter: {adapter_path or 'base model'}")
    print(f"Problems: {num_problems or 'all'}")
    print(f"Samples: {num_samples}")

    # Load dataset
    dataset = load_dataset("nuprl/MultiPL-E", "humaneval-cpp", split="test")
    if num_problems:
        dataset = dataset.select(range(min(num_problems, len(dataset))))

    # Load model
    model_id = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto"
    )

    model_type = "base"
    if adapter_path:
        # Handle relative paths by prepending /data/experiments/
        if not adapter_path.startswith("/"):
            # Remove leading "experiments/" if present since we'll add /data/experiments/
            if adapter_path.startswith("experiments/"):
                adapter_path = adapter_path[len("experiments/"):]
            adapter_path = os.path.join(EXPERIMENTS_DIR, adapter_path)

        if not os.path.exists(adapter_path):
            raise FileNotFoundError(f"Adapter not found: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        model_type = adapter_path.replace("/data/experiments/", "").replace("/", "_")

    model.eval()

    # Generate completions
    completions_data = {
        "metadata": {
            "model_id": model_id,
            "adapter_path": adapter_path,
            "model_type": model_type,
            "num_problems": len(dataset),
            "num_samples": num_samples,
            "temperature": temperature,
            "timestamp": datetime.now().isoformat(),
        },
        "problems": []
    }

    for problem_idx, sample in enumerate(dataset):
        problem_name = sample.get('name', f'problem_{problem_idx}')
        code_prompt = sample['prompt']
        tests = sample.get('tests', '')

        print(f"[{problem_idx+1}/{len(dataset)}] {problem_name}", end=" ")

        problem_data = {
            "name": problem_name,
            "prompt": code_prompt,
            "tests": tests,
            "completions": []
        }

        for sample_idx in range(num_samples):
            messages = [
                {"role": "system", "content": "You are a helpful coding assistant. Please complete the C++ code provided by the user. Output only the valid code completion."},
                {"role": "user", "content": f"Complete the following C++ code:\n\n```cpp\n{code_prompt}\n```"}
            ]

            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=512,
                    temperature=temperature,
                    top_p=0.95,
                    do_sample=temperature > 0
                )

            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            generated_code = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            problem_data["completions"].append({
                "sample_idx": sample_idx,
                "generated_code": generated_code,
            })

        print("✓")
        completions_data["problems"].append(problem_data)

    # Save completions
    os.makedirs(INFERENCE_OUTPUT_DIR, exist_ok=True)

    if run_name:
        filename = f"completions_{run_name}.json"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"completions_{model_type}_{timestamp}.json"

    output_path = os.path.join(INFERENCE_OUTPUT_DIR, filename)
    with open(output_path, "w") as f:
        json.dump(completions_data, f, indent=2)

    print(f"\nCompletions saved to: {output_path}")

    return output_path


@app.function(
    cpu=2,
    timeout=7200,
    volumes={"/data": volume},
    image=eval_image
)
def evaluate_completions(completions_path: str):
    """Evaluate completions (same as eval_completions.py)."""
    import subprocess
    import tempfile
    import math
    import sys
    sys.path.insert(0, "/root")
    from cpp_pipeline.cpp_utils import run_cpp_executable

    print("\n" + "="*80)
    print("STEP 2: EVALUATION")
    print("="*80)
    print(f"Completions: {completions_path}")

    # Load completions
    with open(completions_path, 'r') as f:
        completions_data = json.load(f)

    problems = completions_data["problems"]

    eval_results = {
        "metadata": completions_data["metadata"],
        "problems": [],
        "summary": {}
    }

    total_samples = 0
    total_compiled = 0
    total_runtime_clean = 0
    problems_with_any_pass = 0

    for problem_idx, problem in enumerate(problems):
        problem_name = problem["name"]
        prompt = problem["prompt"]
        tests = problem.get("tests", "")
        completions = problem["completions"]

        print(f"[{problem_idx+1}/{len(problems)}] {problem_name}", end=" ")

        problem_result = {
            "name": problem_name,
            "samples": [],
            "compile_rate": 0.0,
            "runtime_rate": 0.0,
            "passed_any": False
        }

        samples_compiled = 0
        samples_runtime_clean = 0

        for completion in completions:
            sample_idx = completion["sample_idx"]
            generated_code = completion["generated_code"]
            full_code = prompt + generated_code
            if tests:
                full_code += "\n" + tests

            # Compile and run
            result = {"compiled": False, "runtime_clean": False}

            with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
                f.write(full_code)
                src_path = f.name

            exe_path = src_path.replace('.cpp', '.exe')

            try:
                compile_result = subprocess.run(
                    ["g++", "-std=c++17", "-g", "-fsanitize=address", "-fsanitize=undefined",
                     src_path, "-o", exe_path],
                    capture_output=True,
                    text=True,
                    timeout=10
                )

                if compile_result.returncode == 0:
                    result["compiled"] = True
                    samples_compiled += 1
                    total_compiled += 1

                    success, stdout, stderr, runtime_errors = run_cpp_executable(exe_path)
                    if success:
                        result["runtime_clean"] = True
                        samples_runtime_clean += 1
                        total_runtime_clean += 1

            except:
                pass
            finally:
                try:
                    if os.path.exists(src_path):
                        os.unlink(src_path)
                    if os.path.exists(exe_path):
                        os.unlink(exe_path)
                except:
                    pass

            total_samples += 1
            problem_result["samples"].append({
                "sample_idx": sample_idx,
                "compiled": result["compiled"],
                "runtime_clean": result["runtime_clean"],
            })

        problem_result["compile_rate"] = samples_compiled / len(completions) if completions else 0
        problem_result["runtime_rate"] = samples_runtime_clean / len(completions) if completions else 0
        problem_result["passed_any"] = samples_runtime_clean > 0

        if problem_result["passed_any"]:
            problems_with_any_pass += 1

        print(f"Compile: {problem_result['compile_rate']:.0%}, Runtime: {problem_result['runtime_rate']:.0%}")
        eval_results["problems"].append(problem_result)

    # Calculate pass@k
    def calculate_pass_at_k(n: int, c: int, k: int) -> float:
        if n - c < k:
            return 1.0
        return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))

    num_samples_per_problem = completions_data["metadata"].get("num_samples", 10)
    pass_at_k_metrics = {}
    for k in [1, 5, 10]:
        if k <= num_samples_per_problem:
            pass_at_k_values = []
            for problem_result in eval_results["problems"]:
                n = len(problem_result["samples"])
                c = sum(1 for s in problem_result["samples"] if s["runtime_clean"])
                pass_at_k_values.append(calculate_pass_at_k(n, c, k))
            pass_at_k_metrics[f"pass@{k}"] = sum(pass_at_k_values) / len(pass_at_k_values) if pass_at_k_values else 0

    eval_results["summary"] = {
        "total_samples": total_samples,
        "compile_success_rate": total_compiled / total_samples if total_samples > 0 else 0,
        "runtime_success_rate": total_runtime_clean / total_samples if total_samples > 0 else 0,
        "problems_with_any_pass": problems_with_any_pass,
        "problems_with_any_pass_rate": problems_with_any_pass / len(problems) if problems else 0,
        **pass_at_k_metrics
    }

    # Save results
    eval_output_path = completions_path.replace("completions_", "eval_").replace(".json", "_eval.json")
    with open(eval_output_path, "w") as f:
        json.dump(eval_results, f, indent=2)

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"Compile rate: {eval_results['summary']['compile_success_rate']:.2%}")
    print(f"Runtime rate: {eval_results['summary']['runtime_success_rate']:.2%}")
    for k, v in pass_at_k_metrics.items():
        print(f"{k}: {v:.2%}")
    print(f"Results saved to: {eval_output_path}")

    return eval_results


@app.local_entrypoint()
def main(
    adapter_path: Optional[str] = None,
    num_problems: Optional[int] = None,
    num_samples: int = 10,
    temperature: float = 0.8,
    run_name: Optional[str] = None
):
    """
    Run end-to-end evaluation on Modal (inference + evaluation).

    Args:
        adapter_path: Path to LoRA adapter in volume
                     Shorthand: "token-rewards-v1/final_model"
                     Full path: "/data/experiments/token-rewards-v1/final_model"
        num_problems: Number of problems to evaluate (default: all)
        num_samples: Number of samples per problem (default: 10)
        temperature: Sampling temperature (default: 0.8)
        run_name: Name for this run (default: auto-generated)

    Examples:
        # Evaluate base model (quick test with 10 problems)
        modal run training/modal_eval_end_to_end.py --num-problems 10

        # Evaluate base model (all problems)
        modal run training/modal_eval_end_to_end.py

        # Evaluate fine-tuned model (shorthand path)
        modal run training/modal_eval_end_to_end.py --adapter-path token-rewards-v1/final_model

        # Custom configuration
        modal run training/modal_eval_end_to_end.py --adapter-path token-rewards-v1/final_model --num-samples 20 --run-name my_eval
    """
    print("🚀 Starting end-to-end evaluation on Modal...")
    print(f"Adapter: {adapter_path or 'base model'}")
    print(f"Problems: {num_problems or 'all'}")
    print(f"Samples: {num_samples}")

    # Step 1: Generate completions (GPU)
    completions_path = generate_completions.remote(
        adapter_path=adapter_path,
        num_problems=num_problems,
        num_samples=num_samples,
        temperature=temperature,
        run_name=run_name
    )

    # Step 2: Evaluate completions (CPU)
    results = evaluate_completions.remote(completions_path)

    print("\n✅ End-to-end evaluation complete!")
