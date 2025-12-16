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
from cpp_pipeline.cpp_utils import compile_cpp_code, run_cpp_executable, create_token_rewards_from_compiler_errors, link_executable, clean_generated_code
from cpp_pipeline.load_data import get_multipl_e_dataset
from training.config.lora_config import get_unified_lora_config

# Global mapping to store tests for each prompt
PROMPT_TO_TESTS = {}

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
    """Get prompts from MultiPL-E dataset."""
    examples = get_multipl_e_dataset(split="train")
    
    # Populate the global mapping
    global PROMPT_TO_TESTS
    PROMPT_TO_TESTS = {}
    for ex in examples:
        PROMPT_TO_TESTS[ex["prompt"]] = ex["tests"]
        
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
    import logging
    logger = logging.getLogger(__name__)

    rewards_list = []
    batch_size = response_ids.shape[0]
    response_len = response_ids.shape[1]
    device = response_ids.device

    # Track statistics for validation
    batch_stats = {
        "total_samples": batch_size,
        "compile_errors": 0,
        "runtime_errors": 0,
        "clean_code": 0,
        "alignment_success": 0,
        "alignment_failure": 0,
        "total_reward_tokens": 0,
        "error_reward_tokens": 0,
        "runtime_reward_tokens": 0,
        "clean_reward_tokens": 0,
    }

    logger.info(f"[PPO TOKEN REWARDS] Computing for batch_size={batch_size}, response_len={response_len}")

    # Decode prompts and responses
    try:
        prompts = processing_class.batch_decode(query_ids, skip_special_tokens=True)
        responses = processing_class.batch_decode(response_ids, skip_special_tokens=True)
    except Exception as e:
        logger.error(f"[PPO TOKEN REWARDS] FATAL: Failed to decode prompts/responses: {e}")
        # Return default clean rewards to avoid crash
        return torch.ones((batch_size, response_len), dtype=torch.float32, device=device)

    for i in range(batch_size):
        prompt = prompts[i]
        response = responses[i]

        logger.debug(f"[PPO TOKEN REWARDS] Sample {i}: prompt_len={len(prompt)}, response_len={len(response)}")
        
        # Retrieve tests
        tests = PROMPT_TO_TESTS.get(prompt, "")
        
        # Clean response (remove main if present)
        cleaned_response = clean_generated_code(response)
        
        # Handle markdown if present
        if "```cpp" in cleaned_response:
            code_body = cleaned_response.split("```cpp")[1].split("```")[0]
        elif "```c++" in cleaned_response:
            code_body = cleaned_response.split("```c++")[1].split("```")[0]
        elif "```" in cleaned_response:
            code_body = cleaned_response.split("```")[1].split("```")[0]
        else:
            code_body = cleaned_response

        # CRITICAL FIX: The model often generates a complete function body including the closing '}'.
        # The tests from MultiPL-E start with '}' to close the function.
        # If code_body ends with '}', we need to strip it to avoid double closing braces.
        code_body_stripped = code_body.rstrip()
        if code_body_stripped.endswith('}'):
            open_count = code_body_stripped.count('{')
            close_count = code_body_stripped.count('}')
            if close_count > open_count:
                # Model closed the function - remove the trailing }
                last_brace = code_body_stripped.rfind('}')
                code_body = code_body_stripped[:last_brace]

        # Stitch full code
        # prompt ends with {, tests starts with }
        full_code = f"{prompt}\n{code_body}\n{tests}"

        # Compile
        success, stderr, errors = compile_cpp_code(full_code)

        # If compiles, Run
        runtime_errors = []
        if success:
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
                f.write(full_code)
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

        # Compute token rewards for the code_body directly
        # Strategy: Compute rewards for code_body, then align with response tokens
        # This is simpler and more robust than trying to extract from full_code

        # HOWEVER: Error line numbers are relative to full_code, not code_body
        # So we need to adjust error line numbers before passing to reward function

        # Calculate line offset: how many lines before code_body starts?
        prompt_lines = len(prompt.split('\n'))

        # Adjust error line numbers to be relative to code_body
        adjusted_errors = []
        code_body_lines = len(code_body.split('\n'))
        for error in all_errors:
            error_line = error.get('line')
            if error_line is not None:
                # Convert from full_code line number to code_body line number
                adjusted_line = error_line - prompt_lines
                # Only keep errors that fall within code_body range
                if 1 <= adjusted_line <= code_body_lines:
                    adjusted_error = error.copy()
                    adjusted_error['line'] = adjusted_line
                    adjusted_errors.append(adjusted_error)

        # Compute rewards for code_body with adjusted errors
        code_body_rewards = create_token_rewards_from_compiler_errors(
            code_body,
            adjusted_errors,
            processing_class,
            error_reward=0.0,
            runtime_error_reward=0.2,
            clean_reward=1.0
        )
        
        # SAFETY CHECK: If we have errors but rewards are all clean, apply fallback penalty
        has_penalty = any(r < 1.0 for r in code_body_rewards)
        if not has_penalty:
            if not success:
                # Compile failed but no error mapped to body -> Global compile penalty
                code_body_rewards = [0.0] * len(code_body_rewards)
                print(f"WARNING: Compilation failed with unmapped errors. Applying global 0.0 penalty.")
            elif runtime_errors:
                # Runtime failed but no error mapped to body -> Global runtime penalty
                code_body_rewards = [0.2] * len(code_body_rewards)
                print(f"WARNING: Runtime error with unmapped errors. Applying global 0.2 penalty.")

        # Map code_body rewards to response tokens
        seq_rewards = [1.0] * response_len  # Default: clean_reward

        # Find code_body in response (response might have markdown or extra whitespace)
        response_body_start = response.find(code_body)

        if response_body_start != -1 and len(code_body_rewards) > 0:
            # Tokenize response to get offset mapping
            response_encoding = processing_class(response, add_special_tokens=False, return_offsets_mapping=True)
            response_offsets = response_encoding['offset_mapping']

            # Tokenize code_body to get offset mapping
            body_encoding = processing_class(code_body, add_special_tokens=False, return_offsets_mapping=True)
            body_offsets = body_encoding['offset_mapping']

            # Ensure lengths match
            if len(code_body_rewards) != len(body_offsets):
                print(f"WARNING: Reward length mismatch: {len(code_body_rewards)} rewards vs {len(body_offsets)} tokens")
                code_body_rewards = code_body_rewards[:len(body_offsets)]  # Truncate if needed

            # Map code_body tokens to response tokens by character position
            body_end = response_body_start + len(code_body)

            reward_idx = 0
            for resp_idx, (start, end) in enumerate(response_offsets):
                # Check if this response token overlaps with code_body region
                if start >= response_body_start and end <= body_end and resp_idx < response_len:
                    # Find corresponding reward
                    if reward_idx < len(code_body_rewards):
                        seq_rewards[resp_idx] = code_body_rewards[reward_idx]
                        reward_idx += 1
        else:
            # Alignment failed - use fallback strategy
            # If compilation or runtime failed, apply penalty to all tokens
            if not success:
                # Compile error - severe penalty
                seq_rewards = [0.0] * response_len
                print(f"WARNING: Compilation failed, applying 0.0 penalty to all tokens")
            elif runtime_errors:
                # Runtime error - moderate penalty
                seq_rewards = [0.2] * response_len
                print(f"WARNING: Runtime error, applying 0.2 penalty to all tokens")
            else:
                # Clean code - full reward
                seq_rewards = [1.0] * response_len

        # Logging for debugging
        non_default_rewards = [r for r in seq_rewards if r != 1.0]
        if non_default_rewards:
            print(f"  Sample {i}: {len(non_default_rewards)}/{len(seq_rewards)} tokens have non-default rewards")
            print(f"    Compile: {success}, Runtime errors: {len(runtime_errors)}, Adjusted errors in body: {len(adjusted_errors)}")

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
        response_length=512,  # Increased to match inference and handle longer function bodies
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
