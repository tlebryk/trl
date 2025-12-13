import modal
import sys
import os

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
        "tensorboard"
    )
    # Mount local source code into the container
    .add_local_file("dpo_trainer_token_rewards.py", "/root/dpo_trainer_token_rewards.py")
    .add_local_dir("cpp_pipeline", "/root/cpp_pipeline")
    .add_local_file("training/train.py", "/root/train.py")
)

# Define a persistent volume to store checkpoints and logs
# This allows retrieving artifacts after the ephemeral container dies
volume = modal.Volume.from_name("dpo-training-vol", create_if_missing=True)

app = modal.App("token-reward-dpo", image=image)

# Remote output path where results will be written (mounted volume)
REMOTE_OUTPUT_DIR = "/data/results"

@app.function(
    gpu="T4",  # Use T4 GPU as requested (cheap & sufficient for 0.5B model)
    timeout=3600, # 1 hour timeout
    volumes={"/data": volume}, # Mount volume at /data
    image=image  # Use image with mounted local files
)
def train():
    import subprocess
    
    print("Listing remote directory structure:")
    subprocess.run(["ls", "-R", "/root"])

    # Ensure the pipeline data exists (or regenerate it)
    # Since we mounted the directory, the scripts are there. 
    # But we might need to run the pipeline generation first to ensure .json files exist 
    # if we only mounted the python scripts and not the generated data.
    # To be safe, let's run the full pipeline generation inside the container.
    print("Generating training data...")
    subprocess.run(["python", "-m", "cpp_pipeline.run_pipeline"], cwd="/root", check=True)

    # Run the training script
    # Pass the volume path via environment variable
    env = os.environ.copy()
    env["TRAINING_OUTPUT_DIR"] = REMOTE_OUTPUT_DIR
    
    print("Launching training script...")
    subprocess.run(["python", "/root/train.py"], env=env, check=True)
    
    print(f"Training finished. Artifacts stored in volume at {REMOTE_OUTPUT_DIR}")
    
    # List artifacts
    subprocess.run(["ls", "-lh", REMOTE_OUTPUT_DIR])

@app.local_entrypoint()
def main():
    print("Submitting training job to Modal...")
    train.remote()

