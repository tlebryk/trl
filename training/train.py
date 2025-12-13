import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from trl import DPOConfig
from pathlib import Path

# Add project root to path to import local modules
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from dpo_trainer_token_rewards import TokenRewardDPOTrainer
from cpp_pipeline import prepare_dataset

def main():
    # --- Configuration ---
    model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    output_dir = os.environ.get("TRAINING_OUTPUT_DIR", "/results")

    print(f"Starting training run...")
    print(f"Model: {model_name}")
    print(f"Output Directory: {output_dir}")

    # --- Load Data ---
    # In a real scenario, this might load from disk or S3.
    # For now, we regenerate the synthetic dataset on the fly.
    print("Preparing dataset...")
    # Check if running in Modal (files at /root) or locally
    if os.path.exists("/root/cpp_pipeline"):
        cpp_base_dir = "/root/cpp_pipeline"
    else:
        cpp_base_dir = os.path.join(project_root, "cpp_pipeline")
    train_dataset = prepare_dataset(base_dir=cpp_base_dir)
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
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    # We don't need to call get_peft_model explicitly for TRL/DPOTrainer usually, 
    # as passing peft_config to the trainer handles it, but doing it here allows explicit verification.
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

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
        save_strategy="epoch",
        bf16=True, # Use bfloat16
        remove_unused_columns=False,
        report_to="tensorboard",
    )

    # --- Initialize Trainer ---
    print("Initializing TokenRewardDPOTrainer...")
    trainer = TokenRewardDPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        use_token_level_rewards=True # Enable our custom logic
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

