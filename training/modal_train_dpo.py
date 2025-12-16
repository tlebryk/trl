import modal
import os
import json
from datetime import datetime

# Define the image with all necessary dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch",
        "transformers",
        "peft",
        "trl",
        "datasets",
        "accelerate",
        "bitsandbytes",
        "scipy",
        "tensorboard",
    )
    # Mount local source code into the container
    .add_local_file("dpo_trainer_token_rewards.py", "/root/dpo_trainer_token_rewards.py")
    .add_local_dir("cpp_pipeline", "/root/cpp_pipeline")
    .add_local_file("training/train_dpo.py", "/root/train_dpo.py")
    .add_local_dir("training/config", "/root/training/config")
)

# Define a persistent volume to store checkpoints and logs
# This allows retrieving artifacts after the ephemeral container dies
volume = modal.Volume.from_name("dpo-training-vol", create_if_missing=True)

app = modal.App("token-reward-dpo", image=image)

# Remote output path where results will be written (mounted volume)
REMOTE_OUTPUT_DIR_BASE = "/data/experiments"

@app.function(
    gpu="L4",  # Use L4 GPU as requested (cheap & sufficient for 0.5B model)
    timeout=1800, # 30 minutes timeout
    volumes={"/data": volume}, # Mount volume at /data
    image=image  # Use image with mounted local files
)
def train(experiment_name: str, use_token_level_rewards: bool = True):
    import subprocess

    print("Listing remote directory structure:")
    subprocess.run(["ls", "-R", "/root"])

    # Check if training data exists
    # We check multiple possible locations because 'modal volume put' paths can be tricky
    possible_paths = [
        "/data/training_data/dpo_training_data.jsonl",           # Expected path
        "/data/data/training_data/dpo_training_data.jsonl",      # Nested path (common mistake)
        "/root/data/training_data/dpo_training_data.jsonl"       # Fallback
    ]
    
    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            break
            
    if data_path is None:
        print("Could not find training data. Listing /data directory:")
        subprocess.run(["find", "/data", "-maxdepth", "4"])
        raise FileNotFoundError(
            "DPO training data not found. Please ensure it is uploaded to the volume.\n"
            "Try running: modal volume put dpo-training-vol data/training_data/dpo_training_data.jsonl training_data/dpo_training_data.jsonl"
        )
    print(f"✓ Found training data at {data_path}")

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
        "model": "Qwen/Qwen2.5-Coder-0.5B",
        "gpu": "L4",
    }
    metadata_path = os.path.join(output_dir, "run_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")

    # Run the training script
    # Pass config via environment variables
    env = os.environ.copy()
    env["TRAINING_OUTPUT_DIR"] = output_dir
    env["USE_TOKEN_LEVEL_REWARDS"] = str(use_token_level_rewards)
    env["DPO_DATA_PATH"] = data_path

    print("Launching training script...")
    subprocess.run(["python", "/root/train_dpo.py"], env=env, check=True)

    # Commit volume changes
    volume.commit()

    print(f"Training finished. Artifacts stored in volume at {output_dir}")
    print("\nTo download results locally:")
    print(f"  modal volume get dpo-training-vol {output_dir} ./results/{experiment_name}/ --force")

    # List artifacts
    subprocess.run(["ls", "-lh", output_dir])

@app.local_entrypoint()
def main(
    experiment_name: str,
    use_token_level_rewards: bool = True
):
    """
    Submits a DPO training job to Modal.

    Args:
        experiment_name: Name for this experiment (e.g., "baseline-v1", "token-rewards-ablation")
        use_token_level_rewards: Enable token-level rewards (default: True)

    Examples:
        # Run with token-level rewards
        modal run training/modal_train_dpo.py --experiment-name token-rewards-v1

        # Run vanilla DPO baseline
        modal run training/modal_train_dpo.py --experiment-name baseline-dpo --no-use-token-level-rewards

        # Download results after training
        modal volume get dpo-training-vol /experiments/token-rewards-v1 ./results --force
    """
    print("Submitting training job to Modal...")
    print(f"Experiment name: {experiment_name}")
    print(f"Using token-level rewards: {use_token_level_rewards}")

    train.remote(
        experiment_name=experiment_name,
        use_token_level_rewards=use_token_level_rewards
    )
