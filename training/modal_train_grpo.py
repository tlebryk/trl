import modal
import os
import json
from datetime import datetime

# Define the image with all necessary dependencies
# We need g++ and sanitizers for GRPO reward computation (compiling code on the fly)
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("g++", "libasan5", "libubsan1") # Install compiler and sanitizers
    .pip_install(
        "torch",
        "transformers",
        "peft",
        "trl",
        "datasets",
        "accelerate",
        "bitsandbytes",
        "scipy",
        "tensorboard"
    )
    # Mount local source code into the container
    .add_local_file("grpo_trainer_token_rewards.py", "/root/grpo_trainer_token_rewards.py")
    .add_local_dir("cpp_pipeline", "/root/cpp_pipeline")
    .add_local_file("training/train_grpo.py", "/root/train_grpo.py")
    .add_local_dir("training/config", "/root/training/config")
)

# Define a persistent volume to store checkpoints and logs
volume = modal.Volume.from_name("dpo-training-vol", create_if_missing=True)

app = modal.App("token-reward-grpo", image=image)

# Base remote output path where results will be written
REMOTE_OUTPUT_DIR_BASE = "/data/experiments"

@app.function(
    gpu="L4",  # Use L4 GPU
    timeout=7200, # 2 hours timeout (GRPO might be slower due to generation)
    volumes={"/data": volume}, # Mount volume at /data
    image=image  # Use image with mounted local files
)
def train(experiment_name: str, use_token_level_rewards: bool = True):
    import subprocess

    print("Listing remote directory structure:")
    subprocess.run(["ls", "-R", "/root"])

    # Ensure the pipeline data (prompts) exists
    print("Generating example prompts...")
    subprocess.run(["python", "-m", "cpp_pipeline.create_examples"], cwd="/root", check=True)

    # Create experiment directory
    output_dir = os.path.join(REMOTE_OUTPUT_DIR_BASE, experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Experiment: {experiment_name}")
    print(f"Using token-level rewards: {use_token_level_rewards}")
    print(f"Outputting artifacts to: {output_dir}")

    # Save experiment metadata
    metadata = {
        "experiment_name": experiment_name,
        "use_token_level_rewards": use_token_level_rewards,
        "timestamp": datetime.now().isoformat(),
        "model": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        "gpu": "L4",
        "method": "GRPO",
    }
    metadata_path = os.path.join(output_dir, "run_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")

    # Run the training script
    env = os.environ.copy()
    env["TRAINING_OUTPUT_DIR"] = output_dir
    env["USE_TOKEN_LEVEL_REWARDS"] = str(use_token_level_rewards)

    print("Launching GRPO training script...")
    subprocess.run(["python", "/root/train_grpo.py"], env=env, check=True)

    # Commit volume changes
    volume.commit()

    print(f"Training finished. Artifacts stored in volume at {output_dir}")
    print("\nTo download results locally:")
    print(f"  modal volume get dpo-training-vol {output_dir} ./{experiment_name}")

    # List artifacts
    subprocess.run(["ls", "-lh", output_dir])

@app.local_entrypoint()
def main(
    experiment_name: str,
    use_token_level_rewards: bool = True
):
    """
    Submits a GRPO training job to Modal.

    Args:
        experiment_name: Name for this experiment (e.g., "grpo-baseline-v1", "grpo-token-rewards")
        use_token_level_rewards: Enable token-level rewards (default: True)

    Examples:
        # Run with token-level rewards
        modal run training/modal_train_grpo.py --experiment-name grpo-token-rewards-v1

        # Run vanilla GRPO baseline
        modal run training/modal_train_grpo.py --experiment-name grpo-baseline --use-token-level-rewards=false

        # Download results after training
        modal volume get dpo-training-vol /experiments/grpo-token-rewards-v1 ./results
    """
    print("Submitting GRPO training job to Modal...")
    print(f"Experiment name: {experiment_name}")
    print(f"Using token-level rewards: {use_token_level_rewards}")

    train.remote(
        experiment_name=experiment_name,
        use_token_level_rewards=use_token_level_rewards
    )
