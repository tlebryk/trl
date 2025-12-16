import os
import sys
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from datasets import Dataset
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from ppo_trainer_token_rewards import TokenRewardPPOTrainer
from trl.experimental.ppo.ppo_config import PPOConfig
from cpp_pipeline.cpp_utils import compile_cpp_code, run_cpp_executable, create_token_rewards_from_compiler_errors, link_executable
from cpp_pipeline.create_examples import create_example_definitions
from training.config.lora_config import get_unified_lora_config


class LoggingTokenRewardPPOTrainer(TokenRewardPPOTrainer):
    """
    A subclass of TokenRewardPPOTrainer that adds file-based logging for rewards.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_log_path = os.path.join(self.args.output_dir, "ppo_rewards.jsonl")
        # Clear file at the beginning of a run on the main process
        if self.accelerator.is_main_process:
            # Create directory if it doesn't exist
            os.makedirs(self.args.output_dir, exist_ok=True)
            if os.path.exists(self.reward_log_path):
                os.remove(self.reward_log_path)

    def log(self, metrics):
        """
        Overrides the default log method to write reward-related metrics to a JSONL file.
        """
        # Call original log method (for tensorboard etc.)
        super().log(metrics)

        # Also write to our custom file on the main process
        if self.accelerator.is_main_process:
            # Log scalar values that are interesting for reward analysis
            log_metrics = {k: v for k, v in metrics.items() if "reward" in k or "score" in k}
            log_metrics["step"] = self.state.global_step
            try:
                with open(self.reward_log_path, "a") as f:
                    f.write(json.dumps(log_metrics) + "\n")
            except Exception as e:
                print(f"Warning: Failed to write to reward log file: {e}")


def get_prompts():
    """Get prompts from our example definitions."""
    examples = create_example_definitions()
    # PPO expects a "prompt" key in each dataset item
    return [{"prompt": ex["prompt"]} for ex in examples]


def token_reward_function(query_ids, response_ids, response_mask, processing_class):
    """
    Token-level reward function for PPO.

    Args:
        query_ids: (batch, query_len) tensor - the prompts
        response_ids: (batch, response_len) tensor - generated responses
        response_mask: (batch, response_len) tensor - mask for valid response tokens
        processing_class: tokenizer

    Returns:
        rewards: (batch, response_len) tensor - per-token rewards
    """
    rewards_list = []
    batch_size = response_ids.shape[0]
    response_len = response_ids.shape[1]
    device = response_ids.device

    # Decode prompts and responses
    prompts = processing_class.batch_decode(query_ids, skip_special_tokens=True)
    responses = processing_class.batch_decode(response_ids, skip_special_tokens=True)

    for i in range(batch_size):
        prompt = prompts[i]
        response = responses[i]

        # Combine for full code (base model completion)
        full_code = prompt + response

        # Simple extraction heuristic - if markdown is used, trust it
        if "```cpp" in response:
            code = response.split("```cpp")[1].split("```")[0]
        elif "```c++" in response:
            code = response.split("```c++")[1].split("```")[0]
        elif "```" in response:
            code = response.split("```")[1].split("```")[0]
        else:
            code = full_code  # Use full code for compilation

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

        # Compute token rewards for the code
        code_token_rewards = create_token_rewards_from_compiler_errors(
            code,
            all_errors,
            processing_class,
            error_reward=0.0,
            runtime_error_reward=0.2,
            clean_reward=1.0
        )

        # Map rewards to response tokens
        seq_rewards = [0.0] * response_len  # Default neutral

        is_full_code = (code == full_code)

        if is_full_code:
            # We have rewards for the whole sequence (prompt + response)
            # Extract the tail that corresponds to response
            if len(code_token_rewards) >= response_len:
                # Take the last N rewards where N is response length
                relevant_rewards = code_token_rewards[-response_len:]
                seq_rewards = relevant_rewards
            else:
                # Response is longer than code tokens
                start_fill = response_len - len(code_token_rewards)
                for k, r in enumerate(code_token_rewards):
                    seq_rewards[start_fill + k] = r
        else:
            # Code is just a snippet from response (markdown case)
            code_ids = processing_class.encode(code, add_special_tokens=False)
            response_id_list = response_ids[i].tolist()

            # Find start index of code_ids in response_id_list
            start_idx = -1
            n = len(code_ids)
            if n > 0:
                for j in range(len(response_id_list) - n + 1):
                    if response_id_list[j:j+n] == code_ids:
                        start_idx = j
                        break

            if start_idx != -1:
                for k, r in enumerate(code_token_rewards):
                    if start_idx + k < response_len:
                        seq_rewards[start_idx + k] = r
            else:
                # Fallback: fill end
                n_rewards = len(code_token_rewards)
                start = max(0, response_len - n_rewards)
                for k in range(response_len - start):
                    if k < len(code_token_rewards):
                        seq_rewards[start + k] = code_token_rewards[k]

        rewards_list.append(seq_rewards)

    return torch.tensor(rewards_list, dtype=torch.float32, device=device)


def main():
    model_name = "Qwen/Qwen2.5-Coder-0.5B"
    output_dir = os.environ.get("TRAINING_OUTPUT_DIR", "ppo_results")
    use_token_level_rewards_str = os.environ.get("USE_TOKEN_LEVEL_REWARDS", "True")
    use_token_level_rewards = use_token_level_rewards_str.lower() in ("true", "1", "t")

    print(f"Starting PPO training...")
    print(f"Token-level rewards enabled: {use_token_level_rewards}")

    # 1. Load Data (Prompts)
    prompts = get_prompts()
    dataset = Dataset.from_list(prompts)
    print(f"Loaded {len(dataset)} prompts")

    # 2. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine device and dtype
    if torch.cuda.is_available():
        model_dtype = torch.bfloat16
    else:
        model_dtype = torch.float32

    # 3. Load Models
    print("Loading policy model...")
    policy = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=model_dtype,
    )

    print("Loading reference model...")
    ref_policy = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=model_dtype,
    )

    # Value model and Reward model use SequenceClassification head
    # For token-level rewards, the reward_model output is mostly unused
    # but we still need to provide one for the PPO trainer interface
    print("Loading value model...")
    value_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=model_dtype,
        num_labels=1,
    )

    print("Loading reward model (placeholder for token-level rewards)...")
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=model_dtype,
        num_labels=1,
    )

    # 4. PEFT Configuration - use unified config to ensure compatibility
    peft_config = get_unified_lora_config()

    # 5. Prepare dataset
    def prepare_dataset(dataset, tokenizer):
        """Pre-tokenize the dataset before training."""
        def tokenize(element):
            outputs = tokenizer(
                element["prompt"],
                padding=False,
            )
            return {"input_ids": outputs["input_ids"]}

        return dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names,
        )

    train_dataset = prepare_dataset(dataset, tokenizer)
    # Use a small subset for eval
    eval_dataset = train_dataset.select(range(min(10, len(train_dataset))))

    # 6. PPO Configuration
    ppo_config = PPOConfig(
        output_dir=output_dir,
        total_episodes=len(train_dataset) * 2,  # 2 epochs worth
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        logging_steps=1,
        save_steps=10,
        response_length=256,
        num_ppo_epochs=2,
        num_mini_batches=1,
        local_rollout_forward_batch_size=1,
        kl_coef=0.05,
        gamma=1.0,
        lam=0.95,
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=0.1,
        temperature=0.7,
        bf16=True if torch.cuda.is_available() else False,
        seed=42,
        report_to=["tensorboard"],
        num_sample_generations=0,  # Disable sample generations
        sft_model_path=model_name,
        reward_model_path=model_name,
    )

    # 7. Initialize Custom Trainer
    print("Initializing Trainer...")

    # Token reward function (only if enabled)
    token_reward_fn = token_reward_function if use_token_level_rewards else None

    trainer = LoggingTokenRewardPPOTrainer(
        args=ppo_config,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        token_reward_fn=token_reward_fn,
        use_token_level_rewards=use_token_level_rewards,
        combine_with_scalar_reward=False,  # Only use token-level rewards
    )

    # 8. Train
    print("Starting training loop...")
    trainer.train()

    # 9. Save final model
    final_save_path = os.path.join(output_dir, "final_model")
    trainer.save_model(final_save_path)
    print(f"Saved model to {final_save_path}")


if __name__ == "__main__":
    main()
