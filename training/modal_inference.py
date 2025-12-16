"""
Generate code completions using Modal (GPU inference).

This script loads a model (base or fine-tuned with LoRA adapters) and generates
completions for the MultiPL-E humaneval-cpp dataset. Completions are saved to
the Modal volume for later evaluation.

Evaluation (compile/run) happens separately in eval_completions.py.
"""
import modal
import json
import os
from typing import Optional

# Define the image with inference dependencies (no compiler needed)
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch",
        "transformers",
        "peft",
        "datasets",
        "accelerate",
    )
)

# Mount the training volume to access checkpoints
volume = modal.Volume.from_name("dpo-training-vol", create_if_missing=True)

app = modal.App("dpo-inference", image=image)

EXPERIMENTS_DIR = "/data/experiments"
INFERENCE_OUTPUT_DIR = "/data/inference_results"


@app.function(
    gpu="T4",
    timeout=7200,  # 2 hours for full inference run
    volumes={"/data": volume},
    image=image
)
def generate_completions(
    adapter_path: Optional[str] = None,
    num_problems: Optional[int] = None,
    num_samples: int = 10,
    temperature: float = 0.8,
    max_new_tokens: int = 512,
    run_name: Optional[str] = None
):
    """
    Generate code completions for MultiPL-E humaneval-cpp.

    Args:
        adapter_path: Path to LoRA adapter in volume (e.g., "/data/results/token-rewards/final_model")
                     If None, uses base model only.
        num_problems: Number of problems to generate for (None = all)
        num_samples: Number of samples to generate per problem
        temperature: Sampling temperature
        max_new_tokens: Max tokens to generate
        run_name: Name for this run (used in output filename)

    Returns:
        Dict with metadata and path to saved completions
    """
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from datetime import datetime

    print("="*80)
    print("INFERENCE CONFIGURATION")
    print("="*80)
    print(f"Adapter path: {adapter_path or 'None (base model)'}")
    print(f"Num problems: {num_problems or 'All'}")
    print(f"Samples per problem: {num_samples}")
    print(f"Temperature: {temperature}")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Run name: {run_name or 'auto-generated'}")
    print("="*80)

    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset("nuprl/MultiPL-E", "humaneval-cpp", split="test")
    if num_problems:
        dataset = dataset.select(range(min(num_problems, len(dataset))))
    print(f"Generating completions for {len(dataset)} problems")

    # Load model and tokenizer
    model_id = "Qwen/Qwen2.5-Coder-0.5B"
    print(f"\nLoading base model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto"
    )

    # Load adapter if specified
    model_type = "base"
    experiment_name = None
    if adapter_path:
        # Handle relative paths by prepending /data/experiments/
        if not adapter_path.startswith("/"):
            # Remove leading "experiments/" if present since we'll add /data/experiments/
            if adapter_path.startswith("experiments/"):
                adapter_path = adapter_path[len("experiments/"):]
            adapter_path = os.path.join(EXPERIMENTS_DIR, adapter_path)

        print(f"Loading LoRA adapter from: {adapter_path}")

        if not os.path.exists(adapter_path):
            print(f"WARNING: Adapter path {adapter_path} does not exist!")
            print("Available paths in /data/experiments:")
            if os.path.exists(EXPERIMENTS_DIR):
                for root, dirs, files in os.walk(EXPERIMENTS_DIR):
                    print(f"  {root}/")
                    for d in dirs:
                        print(f"    {d}/")
            raise FileNotFoundError(f"Adapter not found at {adapter_path}")

        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()  # Merge for faster inference
        print("Adapter loaded and merged")
        model_type = adapter_path.replace("/data/experiments/", "").replace("/", "_")
        # Extract experiment name (directory containing final_model)
        if "/final_model" in adapter_path:
            experiment_name = adapter_path.replace("/data/experiments/", "").replace("/final_model", "")
        else:
            # Fallback: use parent directory
            experiment_name = os.path.basename(os.path.dirname(adapter_path))

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
            "max_new_tokens": max_new_tokens,
            "timestamp": datetime.now().isoformat(),
        },
        "problems": []
    }

    print("\n" + "="*80)
    print("STARTING INFERENCE")
    print("="*80)

    for problem_idx, sample in enumerate(dataset):
        problem_name = sample.get('name', f'problem_{problem_idx}')
        code_prompt = sample['prompt']
        tests = sample.get('tests', '')

        print(f"\n[{problem_idx+1}/{len(dataset)}] {problem_name}")
        print("-" * 80)

        problem_data = {
            "name": problem_name,
            "prompt": code_prompt,
            "tests": tests,
            "completions": []
        }

        # Generate multiple samples for this problem
        for sample_idx in range(num_samples):
            # Pure code completion: tokenize prompt directly
            model_inputs = tokenizer([code_prompt], return_tensors="pt").to(model.device)

            # Generate
            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=0.95,
                    do_sample=temperature > 0
                )

            # Decode
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            generated_code = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            problem_data["completions"].append({
                "sample_idx": sample_idx,
                "generated_code": generated_code,
            })

            # Progress indicator
            print(f"  Sample {sample_idx+1}/{num_samples}", end="\r")

        print()  # New line after progress
        completions_data["problems"].append(problem_data)

    # Save completions to volume
    # If we have an experiment name, save to experiment directory; otherwise use inference_results
    if experiment_name:
        output_dir = os.path.join(EXPERIMENTS_DIR, experiment_name)
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = INFERENCE_OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)

    if run_name:
        filename = f"completions_{run_name}.json"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"completions_{model_type}_{timestamp}.json"

    output_path = os.path.join(output_dir, filename)

    with open(output_path, "w") as f:
        json.dump(completions_data, f, indent=2)

    # Commit the volume to ensure data is persisted
    volume.commit()

    print("\n" + "="*80)
    print("INFERENCE COMPLETE")
    print("="*80)
    print(f"Generated {len(dataset)} problems × {num_samples} samples = {len(dataset) * num_samples} completions")
    print(f"Saved to: {output_path}")
    print(f"\nTo download for local evaluation:")
    if experiment_name:
        # Modal volume paths use /experiments/ not /data/experiments/
        volume_path = output_path.replace("/data/experiments/", "/experiments/")
        print(f"  modal volume get dpo-training-vol {volume_path} ./results/{experiment_name}/ --force")
    else:
        print(f"  modal volume get dpo-training-vol {output_path[6:]} ./completions/ --force")
    print(f"\nThen run evaluation locally:")
    if experiment_name:
        print(f"  make eval FILE=results/{experiment_name}/{filename} OUTPUT=results/{experiment_name}/eval_{filename}")
    else:
        print(f"  python training/eval_completions.py ./completions/{filename}")
    print("="*80)

    return {
        "output_path": output_path,
        "num_problems": len(dataset),
        "num_samples": num_samples,
        "total_completions": len(dataset) * num_samples
    }


@app.local_entrypoint()
def main(
    adapter_path: Optional[str] = None,
    num_problems: Optional[int] = None,
    num_samples: int = 10,
    temperature: float = 0.8,
    run_name: Optional[str] = None
):
    """
    Run inference on Modal.

    Args:
        adapter_path: Path to LoRA adapter in volume
                     Shorthand: "token-rewards-v1/final_model"
                     Full path: "/data/experiments/token-rewards-v1/final_model"
        num_problems: Number of problems to generate for (default: all)
        num_samples: Number of samples per problem (default: 10)
        temperature: Sampling temperature (default: 0.8)
        run_name: Name for this run (default: auto-generated)

    Examples:
        # Generate completions with base model (10 problems, quick test)
        modal run training/modal_inference.py --num-problems 10

        # Generate completions with base model (all problems)
        modal run training/modal_inference.py

        # Generate completions with fine-tuned model (shorthand path)
        modal run training/modal_inference.py --adapter-path token-rewards-v1/final_model

        # Custom configuration with named run
        modal run training/modal_inference.py --adapter-path token-rewards-v1/final_model --num-samples 20 --temperature 0.6 --run-name my_eval
    """
    print("Submitting inference job to Modal...")
    print(f"Adapter: {adapter_path or 'base model'}")
    print(f"Problems: {num_problems or 'all'}")
    print(f"Samples: {num_samples}")
    print(f"Temperature: {temperature}")
    print(f"Run name: {run_name or 'auto-generated'}")

    result = generate_completions.remote(
        adapter_path=adapter_path,
        num_problems=num_problems,
        num_samples=num_samples,
        temperature=temperature,
        run_name=run_name
    )

    print("\n✅ Inference complete!")
    print(f"Total completions: {result['total_completions']}")
