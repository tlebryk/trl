import modal
import sys
import os

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
)

# Define a persistent volume to store checkpoints and logs
volume = modal.Volume.from_name("dpo-training-vol", create_if_missing=True)

app = modal.App("token-reward-grpo", image=image)

# Base remote output path where results will be written
REMOTE_OUTPUT_DIR_BASE = "/data/results"

@app.function(
    gpu="T4",  # Use T4 GPU
    timeout=7200, # 2 hours timeout (GRPO might be slower due to generation)
    volumes={"/data": volume}, # Mount volume at /data
    image=image  # Use image with mounted local files
)
def train(use_token_level_rewards: bool = True):
    import subprocess

    print("Listing remote directory structure:")
    subprocess.run(["ls", "-R", "/root"])

    # Ensure the pipeline data (prompts) exists
    print("Generating example prompts...")
    subprocess.run(["python", "-m", "cpp_pipeline.create_examples"], cwd="/root", check=True)

    # Determine run type and output directory
    run_type = "grpo-token-rewards" if use_token_level_rewards else "grpo-vanilla"
    output_dir = os.path.join(REMOTE_OUTPUT_DIR_BASE, run_type)
    print(f"Run type: {run_type}")
    print(f"Outputting artifacts to: {output_dir}")

    # Run the training script
    env = os.environ.copy()
    env["TRAINING_OUTPUT_DIR"] = output_dir
    env["USE_TOKEN_LEVEL_REWARDS"] = str(use_token_level_rewards)

    print("Launching GRPO training script...")
    subprocess.run(["python", "/root/train_grpo.py"], env=env, check=True)

    print(f"Training finished. Artifacts stored in volume at {output_dir}")

    # List artifacts
    subprocess.run(["ls", "-lh", output_dir])

@app.local_entrypoint()
def main():
    """
    Submits a GRPO training job to Modal.

    By default, it runs with token-level rewards.
    Use --no-token-rewards to run a baseline vanilla GRPO training.

    Examples:
    # Run with token-level rewards
    modal run training/modal_train_grpo.py

    # Run vanilla GRPO
    modal run training/modal_train_grpo.py --no-token-rewards
    """
    use_token_rewards = "--no-token-rewards" not in sys.argv

    print("Submitting GRPO training job to Modal...")
    print(f"Using token-level rewards: {use_token_rewards}")
    train.remote(use_token_level_rewards=use_token_rewards)
