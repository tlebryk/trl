import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType
from trl.experimental.ppo import PPOConfig
from datasets import Dataset
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from ppo_trainer_token_rewards import TokenRewardPPOTrainer
from cpp_pipeline.cpp_utils import compile_cpp_code, run_cpp_executable, create_token_rewards_from_compiler_errors, link_executable
from cpp_pipeline.create_examples import create_example_definitions

def get_prompts():
    """Get prompts from our example definitions."""
    examples = create_example_definitions()
    return [{"query": ex["prompt"]} for ex in examples]

def main():
    model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    output_dir = os.environ.get("TRAINING_OUTPUT_DIR", "ppo_results")
    
    print(f"Starting PPO training...")
    
    # 1. Load Data (Prompts)
    prompts = get_prompts()
    dataset = Dataset.from_list(prompts)
    print(f"Loaded {len(dataset)} prompts")

    # 2. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(sample):
        return tokenizer(sample["query"], padding="max_length", max_length=64, truncation=True)
    
    dataset = dataset.map(tokenize, batched=True, remove_columns=["query"])
    dataset.set_format(type="torch")

    # Determine device and dtype
    if torch.cuda.is_available():
        device_map = "auto"
        model_dtype = torch.bfloat16
    else:
        # Fallback to CPU/Float32 for local testing to avoid MPS BFloat16 issues
        device_map = "cpu"
        model_dtype = torch.float32

    # 3. Load Policy Model
    print("Loading Policy Model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=model_dtype,
        trust_remote_code=True,
        device_map=device_map
    )

    # 4. Load Value Model
    print("Loading Value Model...")
    value_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        trust_remote_code=True,
        dtype=model_dtype,
        device_map=device_map
    )
    # Value model needs to handle the same inputs
    value_model.config.pad_token_id = tokenizer.pad_token_id
    
    # PEFT for Policy
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # 5. Define Token Reward Function
    def token_reward_fn(input_ids, processing_class):
        """
        Args:
            input_ids: (batch, seq_len) tensor of full sequences (query + response)
            processing_class: tokenizer
        Returns:
            rewards: (batch, seq_len) tensor
        """
        rewards_list = []
        batch_size = input_ids.shape[0]
        
        # Decode batch
        texts = processing_class.batch_decode(input_ids, skip_special_tokens=True)
        
        for i in range(batch_size):
            text = texts[i]
            
            # Simple extraction heuristic
            if "```cpp" in text:
                code = text.split("```cpp")[1].split("```")[0]
            elif "```c++" in text:
                code = text.split("```c++")[1].split("```")[0]
            elif "```" in text:
                code = text.split("```")[1].split("```")[0]
            else:
                code = text # Attempt to compile everything
                
            # Compile
            success, stderr, errors = compile_cpp_code(code)
            
            # If compiles, Run
            runtime_errors = []
            if success:
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
                    f.write(code)
                    src_path = f.name
                
                exe_path = src_path + ".exe"
                if link_executable(src_path, exe_path):
                    run_success, _, _, run_errors = run_cpp_executable(exe_path)
                    if not run_success:
                        runtime_errors = run_errors
                    
                    if os.path.exists(exe_path):
                        os.unlink(exe_path)
                
                if os.path.exists(src_path):
                    os.unlink(src_path)
            
            all_errors = errors + runtime_errors
            
            # Compute rewards for the extracted code
            code_token_rewards = create_token_rewards_from_compiler_errors(
                code, 
                all_errors, 
                processing_class,
                error_reward=0.0,
                runtime_error_reward=0.2,
                clean_reward=1.0
            )
            
            full_len = input_ids.shape[1]
            seq_rewards = [0.0] * full_len # Default neutral
            
            code_ids = processing_class.encode(code, add_special_tokens=False)
            input_id_list = input_ids[i].tolist()
            
            # Find start index of code_ids in input_id_list
            start_idx = -1
            n = len(code_ids)
            for j in range(len(input_id_list) - n + 1):
                # Rough matching (exact match might fail due to spacing/tokenization diffs in full context)
                if input_id_list[j:j+n] == code_ids:
                    start_idx = j
                    break
            
            if start_idx != -1:
                for k, r in enumerate(code_token_rewards):
                    if start_idx + k < full_len:
                        seq_rewards[start_idx + k] = r
            else:
                # Fallback: fill end
                n_rewards = len(code_token_rewards)
                start = max(0, full_len - n_rewards)
                for k in range(full_len - start):
                    seq_rewards[start + k] = code_token_rewards[k]
            
            rewards_list.append(seq_rewards)
            
        return torch.tensor(rewards_list, dtype=torch.float32).to(input_ids.device)

    # 6. Configuration
    ppo_config = PPOConfig(
        exp_name="ppo_token_rewards",
        learning_rate=1e-5,
        mini_batch_size=2,
        batch_size=4,
        gradient_accumulation_steps=1,
        num_ppo_epochs=2,
        num_total_batches=5,  # Quick trial: 5 batches x 4 samples = 20 generations total
        response_length=128,  # Allow 128 tokens for C++ code generation
        num_sample_generations=0,  # Disable sample generation logging (no eval dataset)
        seed=42,
        bf16=True,  # Enable bf16 on T4 GPU for speed
        fp16=False,
    )
    # Output dir is handled by TrainingArguments usually, but PPOConfig inherits it.
    ppo_config.output_dir = output_dir

    # 7. Initialize Custom Trainer
    print("Initializing TokenRewardPPOTrainer...")
    trainer = TokenRewardPPOTrainer(
        args=ppo_config,
        processing_class=tokenizer,
        model=model,
        ref_model=None, # Will use PEFT
        reward_model=value_model, # Reuse value model structure but we ignore its scores for rewards
        value_model=value_model,
        train_dataset=dataset,
        data_collator=None,
        peft_config=peft_config,
        token_reward_fn=token_reward_fn,
    )

    # 8. Train
    print("Starting training loop...")
    trainer.train()
    
    # 9. Save
    trainer.save_model(output_dir)
    print(f"Saved model to {output_dir}")

if __name__ == "__main__":
    main()
