import os
import sys
import torch
import json
from transformers import AutoTokenizer
from trl.trainer.grpo_config import GRPOConfig
from datasets import Dataset
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from grpo_trainer_token_rewards import TokenRewardGRPOTrainer
from cpp_pipeline.cpp_utils import compile_cpp_code, run_cpp_executable, create_token_rewards_from_compiler_errors, link_executable
from cpp_pipeline.create_examples import create_example_definitions
from training.config.lora_config import get_unified_lora_config

class LoggingTokenRewardGRPOTrainer(TokenRewardGRPOTrainer):
    """
    A subclass of TokenRewardGRPOTrainer that adds file-based logging for rewards.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_log_path = os.path.join(self.args.output_dir, "grpo_rewards.jsonl")
        # Clear file at the beginning of a run on the main process
        if self.accelerator.is_main_process:
            # Create directory if it doesn't exist
            os.makedirs(self.args.output_dir, exist_ok=True)
            if os.path.exists(self.reward_log_path):
                os.remove(self.reward_log_path)

    def log(self, metrics, start_time=None):
        """
        Overrides the default log method to write reward-related metrics to a JSONL file.
        """
        # Call original log method (for tensorboard etc.)
        super().log(metrics, start_time)

        # Also write to our custom file on the main process
        if self.accelerator.is_main_process:
            # Log scalar values that are interesting for reward analysis
            log_metrics = {k: v for k, v in metrics.items() if "reward" in k or "advantage" in k}
            log_metrics["step"] = self.state.global_step
            try:
                with open(self.reward_log_path, "a") as f:
                    f.write(json.dumps(log_metrics) + "\n")
            except Exception as e:
                print(f"Warning: Failed to write to reward log file: {e}")

def get_prompts():
    """Get prompts from our example definitions."""
    examples = create_example_definitions()
    # GRPO expects a "prompt" key in each dataset item
    return [{"prompt": ex["prompt"]} for ex in examples]

def cpp_compiler_reward_function(prompts, completions, completion_ids, **kwargs):
    """
    Scalar reward function for GRPO that compiles and runs C++ code.

    Args:
        prompts: List of prompt strings
        completions: List of completion strings (generated text)
        completion_ids: List of completion token IDs
        **kwargs: Additional arguments (ignored)

    Returns:
        List of scalar rewards (float)
    """
    rewards = []

    for prompt, completion in zip(prompts, completions):
        # Extract code from completion
        text = completion
        if isinstance(completion, list):
            # Handle conversational format
            text = completion[0]["content"] if len(completion) > 0 else ""

        # Simple extraction heuristic
        if "```cpp" in text:
            code = text.split("```cpp")[1].split("```")[0]
        elif "```c++" in text:
            code = text.split("```c++")[1].split("```")[0]
        elif "```" in text:
            code = text.split("```")[1].split("```")[0]
        else:
            code = text  # Attempt to compile everything

        # Compile
        success, stderr, errors = compile_cpp_code(code)

        # Compute scalar reward based on compilation and runtime
        if not success:
            # Compilation failed
            reward = 0.0
        else:
            # Compilation succeeded, try to run
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
                f.write(code)
                src_path = f.name

            exe_path = src_path + ".exe"
            runtime_success = False
            if link_executable(src_path, exe_path):
                run_success, _, _, run_errors = run_cpp_executable(exe_path)
                runtime_success = run_success

                if os.path.exists(exe_path):
                    os.unlink(exe_path)

            if os.path.exists(src_path):
                os.unlink(src_path)

            if runtime_success:
                # Clean execution
                reward = 1.0
            else:
                # Runtime error
                reward = 0.2

        rewards.append(reward)

    return rewards

def main():
    model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    output_dir = os.environ.get("TRAINING_OUTPUT_DIR", "grpo_results")
    use_token_level_rewards_str = os.environ.get("USE_TOKEN_LEVEL_REWARDS", "True")
    use_token_level_rewards = use_token_level_rewards_str.lower() in ("true", "1", "t")

    print(f"Starting GRPO training...")
    print(f"Token-level rewards enabled: {use_token_level_rewards}")

    # 1. Load Data (Prompts)
    prompts = get_prompts()
    dataset = Dataset.from_list(prompts)
    print(f"Loaded {len(dataset)} prompts")

    # 2. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine device and dtype
    if torch.cuda.is_available():
        model_dtype = torch.bfloat16
    else:
        # Fallback to Float32 for local testing
        model_dtype = torch.float32

    # 3. Define Token Reward Function (only if needed)
    token_reward_fn = None
    if use_token_level_rewards:
        def token_reward_function(prompt_ids, completion_ids, completion_mask, processing_class):
            """
            Token-level reward function for GRPO.

            Args:
                prompt_ids: (batch, prompt_len) tensor
                completion_ids: (batch, completion_len) tensor
                completion_mask: (batch, completion_len) tensor
                processing_class: tokenizer

            Returns:
                rewards: (batch, completion_len) tensor
            """
            rewards_list = []
            batch_size = completion_ids.shape[0]

            # Decode completions
            completions = processing_class.batch_decode(completion_ids, skip_special_tokens=True)

            for i in range(batch_size):
                text = completions[i]

                # Simple extraction heuristic
                if "```cpp" in text:
                    code = text.split("```cpp")[1].split("```")[0]
                elif "```c++" in text:
                    code = text.split("```c++")[1].split("```")[0]
                elif "```" in text:
                    code = text.split("```")[1].split("```")[0]
                else:
                    code = text  # Attempt to compile everything

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

                # Map to completion tokens
                completion_len = completion_ids.shape[1]
                seq_rewards = [0.0] * completion_len  # Default neutral

                code_ids = processing_class.encode(code, add_special_tokens=False)
                completion_id_list = completion_ids[i].tolist()

                # Find start index of code_ids in completion_id_list
                start_idx = -1
                n = len(code_ids)
                for j in range(len(completion_id_list) - n + 1):
                    if completion_id_list[j:j+n] == code_ids:
                        start_idx = j
                        break

                if start_idx != -1:
                    for k, r in enumerate(code_token_rewards):
                        if start_idx + k < completion_len:
                            seq_rewards[start_idx + k] = r
                else:
                    # Fallback: fill end
                    n_rewards = len(code_token_rewards)
                    start = max(0, completion_len - n_rewards)
                    for k in range(completion_len - start):
                        if k < len(code_token_rewards):
                            seq_rewards[start + k] = code_token_rewards[k]

                rewards_list.append(seq_rewards)

            return torch.tensor(rewards_list, dtype=torch.float32).to(completion_ids.device)

        token_reward_fn = token_reward_function

    # 4. PEFT Configuration - use unified config to ensure compatibility with DPO adapters
    peft_config = get_unified_lora_config()

    # 5. GRPO Configuration
    grpo_config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,  # generation_batch_size = 1*4=4, divisible by num_generations=4
        learning_rate=1e-5,
        logging_steps=1,
        save_steps=10,
        max_prompt_length=128,
        max_completion_length=256,
        num_generations=4,  # Generate 4 completions per prompt for group normalization
        temperature=0.7,
        bf16=True if torch.cuda.is_available() else False,
        fp16=False,
        seed=42,
        report_to=["tensorboard"],
        scale_rewards="group",  # Group-wise reward normalization
        save_strategy="no",  # Don't save intermediate checkpoints
        save_only_model=True,  # Skip optimizer/scheduler states (saves space)
    )

    # 6. Initialize Custom Trainer
    print("Initializing Trainer...")
    trainer = LoggingTokenRewardGRPOTrainer(
        model=model_name,
        reward_funcs=cpp_compiler_reward_function,  # Scalar reward function for baseline
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        token_reward_fn=token_reward_fn,  # Token-level reward function
        gamma=1.0,  # No discounting
        use_token_level_rewards=use_token_level_rewards,
    )

    # 7. Train
    print("Starting training loop...")
    trainer.train()

    # 8. Save final model
    final_save_path = os.path.join(output_dir, "final_model")
    trainer.save_model(final_save_path)
    print(f"Saved model to {final_save_path}")

if __name__ == "__main__":
    main()
