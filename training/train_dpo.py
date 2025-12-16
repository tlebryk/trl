import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig
from pathlib import Path

# Add project root to path to import local modules
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from dpo_trainer_token_rewards import TokenRewardDPOTrainer
from cpp_pipeline.load_data import get_dpo_dataset
from training.config.lora_config import get_unified_lora_config

def main():
    # --- Configuration ---
    model_name = "Qwen/Qwen2.5-Coder-0.5B"
    output_dir = os.environ.get("TRAINING_OUTPUT_DIR", "/results")

    print(f"Starting training run...")
    print(f"Model: {model_name}")
    print(f"Output Directory: {output_dir}")

    # --- Load Data ---
    print("Preparing dataset...")
    # Determine data path (local vs Modal)
    # Check environment variable first, then volume, then root, then local
    data_path = os.environ.get("DPO_DATA_PATH")
    
    if not data_path:
        if os.path.exists("/data/training_data/dpo_training_data.jsonl"):
            data_path = "/data/training_data/dpo_training_data.jsonl"
        elif os.path.exists("/data/data/training_data/dpo_training_data.jsonl"):
            data_path = "/data/data/training_data/dpo_training_data.jsonl"
        elif os.path.exists("/root/data/training_data/dpo_training_data.jsonl"):
            data_path = "/root/data/training_data/dpo_training_data.jsonl"
        else:
            data_path = "data/training_data/dpo_training_data.jsonl"
    
    print(f"Loading dataset from: {data_path}")
    train_dataset = get_dpo_dataset(data_path)
    print(f"Dataset size: {len(train_dataset)}")

    # --- Load Model & Tokenizer ---
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    # For 0.5B model, we can load in bfloat16 easily on T4
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )

    # --- Configure LoRA ---
    print("Configuring LoRA...")
    # Use unified LoRA config to ensure compatibility with PPO adapters
    peft_config = get_unified_lora_config()

    # --- Training Config ---
    training_args = DPOConfig(
        output_dir=output_dir,
        beta=0.1,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        per_device_train_batch_size=2, # Small batch size for T4 compatibility
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        logging_steps=1,
        save_strategy="no",  # Don't save intermediate checkpoints
        save_only_model=True,  # Skip optimizer/scheduler states (saves space)
        bf16=True, # Use bfloat16
        remove_unused_columns=False,
        report_to="tensorboard",
    )

    # --- Determine reward strategy from environment variable ---
    use_token_level_rewards_str = os.environ.get("USE_TOKEN_LEVEL_REWARDS", "True")
    use_token_level_rewards = use_token_level_rewards_str.lower() in ("true", "1", "t")

    # --- Initialize Trainer ---
    print(f"Initializing TokenRewardDPOTrainer (use_token_level_rewards={use_token_level_rewards})...")
    trainer = TokenRewardDPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        use_token_level_rewards=use_token_level_rewards,
        peft_config=peft_config,  # Pass LoRA config to trainer
    )

    # --- Train ---
    print("Starting training...")
    trainer.train()

    # --- Save ---
    print("Saving model...")
    final_save_path = os.path.join(output_dir, "final_model")
    trainer.save_model(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    print(f"Training complete. Model saved to {final_save_path}")

if __name__ == "__main__":
    main()

