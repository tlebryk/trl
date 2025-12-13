import modal
import sys
import os

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
)

# Define a persistent volume to store checkpoints and logs
volume = modal.Volume.from_name("dpo-training-vol", create_if_missing=True)

app = modal.App("token-reward-ppo", image=image)

# Remote output path where results will be written (mounted volume)
REMOTE_OUTPUT_DIR = "/data/ppo_results"

@app.function(
    gpu="T4",  # Use T4 GPU
    timeout=7200, # 2 hours timeout (PPO might be slower due to generation)
    volumes={"/data": volume}, # Mount volume at /data
    image=image  # Use image with mounted local files
)
def train():
    import subprocess
    
    print("Listing remote directory structure:")
    subprocess.run(["ls", "-R", "/root"])

    # Ensure the pipeline data (prompts) exists
    # Even for PPO, we use the prompts from our examples
    print("Generating example prompts...")
    subprocess.run(["python", "-m", "cpp_pipeline.create_examples"], cwd="/root", check=True)

    # Run the training script
    env = os.environ.copy()
    env["TRAINING_OUTPUT_DIR"] = REMOTE_OUTPUT_DIR
    
    print("Launching PPO training script...")
    # This script will compile and run code inside the container
    subprocess.run(["python", "/root/train_ppo.py"], env=env, check=True)
    
    print(f"Training finished. Artifacts stored in volume at {REMOTE_OUTPUT_DIR}")
    
    # List artifacts
    subprocess.run(["ls", "-lh", REMOTE_OUTPUT_DIR])

@app.local_entrypoint()
def main():
    print("Submitting PPO training job to Modal...")
    train.remote()

