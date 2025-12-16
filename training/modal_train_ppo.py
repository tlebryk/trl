import modal
import os
import sys
import json
from datetime import datetime

# Define the image with all necessary dependencies
# We need g++ and sanitizers for PPO reward computation (compiling code on the fly)
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
    .add_local_file("ppo_trainer_token_rewards.py", "/root/ppo_trainer_token_rewards.py")
    .add_local_dir("cpp_pipeline", "/root/cpp_pipeline")
    .add_local_file("training/train_ppo.py", "/root/train_ppo.py")
    .add_local_dir("training/config", "/root/training/config")
)

# Define a persistent volume to store checkpoints and logs
volume = modal.Volume.from_name("dpo-training-vol", create_if_missing=True)

app = modal.App("token-reward-ppo", image=image)

# Base remote output path where results will be written
REMOTE_OUTPUT_DIR_BASE = "/data/experiments"

@app.function(
    gpu="T4",  # Use T4 GPU
    timeout=7200, # 2 hours timeout (PPO might be slower due to generation)
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
        "gpu": "T4",
        "method": "PPO",
    }
    metadata_path = os.path.join(output_dir, "run_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")

    # Run the training script
    env = os.environ.copy()
    env["TRAINING_OUTPUT_DIR"] = output_dir
    env["USE_TOKEN_LEVEL_REWARDS"] = str(use_token_level_rewards)

    print("Launching PPO training script...")
    result = subprocess.run(
        ["python", "/root/train_ppo.py"],
        env=env,
        capture_output=True,
        text=True
    )

    # Print output (helps debug issues)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr, file=sys.stderr)

    # Check for errors
    if result.returncode != 0:
        raise RuntimeError(f"Training script failed with exit code {result.returncode}")

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
    Submits a PPO training job to Modal.

    Args:
        experiment_name: Name for this experiment (e.g., "ppo-baseline-v1", "ppo-token-rewards")
        use_token_level_rewards: Enable token-level rewards (default: True)

    Examples:
        # Run with token-level rewards
        modal run training/modal_train_ppo.py --experiment-name ppo-token-rewards-v1

        # Run vanilla PPO baseline
        modal run training/modal_train_ppo.py --experiment-name ppo-baseline --use-token-level-rewards=false

        # Download results after training
        modal volume get dpo-training-vol /experiments/ppo-token-rewards-v1 ./results
    """
    print("Submitting PPO training job to Modal...")
    print(f"Experiment name: {experiment_name}")
    print(f"Using token-level rewards: {use_token_level_rewards}")

    train.remote(
        experiment_name=experiment_name,
        use_token_level_rewards=use_token_level_rewards
    )
