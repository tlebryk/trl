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

    # --- Determine reward strategy from environment variable ---
    use_token_level_rewards_str = os.environ.get("USE_TOKEN_LEVEL_REWARDS", "True")
    use_token_level_rewards = use_token_level_rewards_str.lower() in ("true", "1", "t")
    
    # Debug mode for reward analysis
    debug_reward_analysis_str = os.environ.get("DEBUG_REWARD_ANALYSIS", "True")
    debug_reward_analysis = debug_reward_analysis_str.lower() in ("true", "1", "t")
    
    print(f"Token-level rewards: {use_token_level_rewards}")
    print(f"Debug reward analysis: {debug_reward_analysis}")

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
    
    # Comprehensive dataset analysis (if debug mode enabled)
    dataset_stats = {
        "total": len(train_dataset),
        "has_chosen_rewards": 0,
        "has_rejected_rewards": 0,
        "chosen_uniform": 0,
        "chosen_non_uniform": 0,
        "rejected_uniform": 0,
        "rejected_non_uniform": 0,
        "both_uniform": 0,
        "both_non_uniform": 0,
        "mixed": 0,
    }
    
    # Verify token rewards are present in dataset
    if use_token_level_rewards and len(train_dataset) > 0:
        sample = train_dataset[0]
        has_chosen_rewards = "chosen_token_rewards" in sample
        has_rejected_rewards = "rejected_token_rewards" in sample
        
        if has_chosen_rewards and has_rejected_rewards:
            print(f"✓ Token-level rewards detected in dataset!")
            
            # Analyze full dataset if debug mode
            if debug_reward_analysis:
                print(f"\n[PRE-TRAINING ANALYSIS] Analyzing {len(train_dataset)} examples...")
                for i, example in enumerate(train_dataset):
                    chosen_rewards = example.get("chosen_token_rewards", [])
                    rejected_rewards = example.get("rejected_token_rewards", [])
                    
                    if chosen_rewards:
                        dataset_stats["has_chosen_rewards"] += 1
                        chosen_is_uniform = len(set(chosen_rewards)) == 1
                        if chosen_is_uniform:
                            dataset_stats["chosen_uniform"] += 1
                        else:
                            dataset_stats["chosen_non_uniform"] += 1
                    
                    if rejected_rewards:
                        dataset_stats["has_rejected_rewards"] += 1
                        rejected_is_uniform = len(set(rejected_rewards)) == 1
                        if rejected_is_uniform:
                            dataset_stats["rejected_uniform"] += 1
                        else:
                            dataset_stats["rejected_non_uniform"] += 1
                    
                    if chosen_rewards and rejected_rewards:
                        if chosen_is_uniform and rejected_is_uniform:
                            dataset_stats["both_uniform"] += 1
                        elif not chosen_is_uniform and not rejected_is_uniform:
                            dataset_stats["both_non_uniform"] += 1
                        else:
                            dataset_stats["mixed"] += 1
                
                # Print comprehensive summary
                print("\n" + "="*70)
                print("PRE-TRAINING DATASET REWARD ANALYSIS")
                print("="*70)
                print(f"Total examples: {dataset_stats['total']}")
                print()
                print("CHOSEN REWARDS:")
                print(f"  Examples with rewards:  {dataset_stats['has_chosen_rewards']:5d}")
                print(f"  Uniform (all same):     {dataset_stats['chosen_uniform']:5d} ({100*dataset_stats['chosen_uniform']/max(dataset_stats['has_chosen_rewards'],1):.1f}%)")
                print(f"  Non-uniform (varied):   {dataset_stats['chosen_non_uniform']:5d} ({100*dataset_stats['chosen_non_uniform']/max(dataset_stats['has_chosen_rewards'],1):.1f}%)")
                print()
                print("REJECTED REWARDS:")
                print(f"  Examples with rewards:  {dataset_stats['has_rejected_rewards']:5d}")
                print(f"  Uniform (all same):      {dataset_stats['rejected_uniform']:5d} ({100*dataset_stats['rejected_uniform']/max(dataset_stats['has_rejected_rewards'],1):.1f}%)")
                print(f"  Non-uniform (varied):   {dataset_stats['rejected_non_uniform']:5d} ({100*dataset_stats['rejected_non_uniform']/max(dataset_stats['has_rejected_rewards'],1):.1f}%)")
                print()
                print("COMBINED PATTERNS:")
                total_paired = dataset_stats['both_uniform'] + dataset_stats['both_non_uniform'] + dataset_stats['mixed']
                if total_paired > 0:
                    print(f"  Both uniform:            {dataset_stats['both_uniform']:5d} ({100*dataset_stats['both_uniform']/total_paired:.1f}%)")
                    print(f"  Both non-uniform:        {dataset_stats['both_non_uniform']:5d} ({100*dataset_stats['both_non_uniform']/total_paired:.1f}%)")
                    print(f"  Mixed (one uniform):     {dataset_stats['mixed']:5d} ({100*dataset_stats['mixed']/total_paired:.1f}%)")
                
                # Health check
                if dataset_stats['both_uniform'] == total_paired:
                    print("\n⚠️  WARNING: ALL examples have uniform rewards! Token-level training may not be effective.")
                elif dataset_stats['both_uniform'] > total_paired * 0.5:
                    print("\n⚠️  WARNING: >50% examples have uniform rewards. Consider checking reward computation.")
                elif dataset_stats['both_non_uniform'] > total_paired * 0.3:
                    print("\n✓ GOOD: Significant portion of examples have non-uniform rewards.")
                else:
                    print("\n✓ OK: Some examples have non-uniform rewards.")
                print("="*70 + "\n")
            else:
                # Quick sample check
                chosen_rewards = sample["chosen_token_rewards"]
                rejected_rewards = sample["rejected_token_rewards"]
                print(f"  Sample chosen rewards: {len(chosen_rewards)} tokens, min={min(chosen_rewards):.2f}, max={max(chosen_rewards):.2f}, mean={sum(chosen_rewards)/len(chosen_rewards):.2f}")
                print(f"  Sample rejected rewards: {len(rejected_rewards)} tokens, min={min(rejected_rewards):.2f}, max={max(rejected_rewards):.2f}, mean={sum(rejected_rewards)/len(rejected_rewards):.2f}")
                
                chosen_unique = set(chosen_rewards)
                rejected_unique = set(rejected_rewards)
                print(f"  Chosen unique rewards: {sorted(chosen_unique)}")
                print(f"  Rejected unique rewards: {sorted(rejected_unique)}")
                
                if len(chosen_unique) == 1 and len(rejected_unique) == 1 and chosen_unique == rejected_unique:
                    print("  ⚠️  WARNING: Sample has uniform rewards! Check full dataset.")
                else:
                    print("  ✓ Sample shows disparate rewards.")
        else:
            print("⚠️  WARNING: Token-level rewards requested but not found in dataset!")
            print(f"   Available keys: {list(sample.keys())}")
            print("   Training will fall back to standard DPO (all rewards = 1.0)")

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

    # --- Initialize Trainer ---
    print(f"Initializing TokenRewardDPOTrainer (use_token_level_rewards={use_token_level_rewards})...")
    trainer = TokenRewardDPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        use_token_level_rewards=use_token_level_rewards,
        debug_reward_analysis=debug_reward_analysis,
        peft_config=peft_config,  # Pass LoRA config to trainer
    )

    # --- Train ---
    print("Starting training...")
    trainer.train()
    
    # Print reward analysis summary after training
    if use_token_level_rewards and debug_reward_analysis:
        trainer.print_reward_analysis_summary()

    # --- Save ---
    print("Saving model...")
    final_save_path = os.path.join(output_dir, "final_model")
    trainer.save_model(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    print(f"Training complete. Model saved to {final_save_path}")

if __name__ == "__main__":
    main()

