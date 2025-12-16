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

        # Compute token rewards for the code
        # Note: errors line numbers refer to full_code.
        # We need to map them back to response tokens.
        # This is tricky because full_code is constructed from prompt + code_body + tests.
        # create_token_rewards_from_compiler_errors expects code and errors.
        # It assigns rewards to lines.
        # We need to pass the code that matches the lines.
        
        # Strategy:
        # 1. Compute rewards for full_code lines.
        # 2. Identify which lines belong to code_body.
        # 3. Assign rewards to code_body tokens.
        
        # However, create_token_rewards_from_compiler_errors assumes it's tokenizing 'code'.
        # If we pass full_code, it will tokenize full_code.
        # We want rewards for response_ids.
        
        # Let's simplify: 
        # We only care about errors in the generated part.
        # If error is in prompt or tests, it's not the model's fault (mostly), or it's context.
        # We can pass full_code to create_token_rewards...
        # And then extract the part corresponding to code_body.
        
        code_token_rewards = create_token_rewards_from_compiler_errors(
            full_code,
            all_errors,
            processing_class,
            error_reward=0.0,
            runtime_error_reward=0.2,
            clean_reward=1.0
        )

        # Map rewards to response tokens
        seq_rewards = [0.0] * response_len  # Default neutral

        # We need to align full_code tokens with response tokens.
        # response contains code_body (maybe with some extras/markdown).
        # full_code contains prompt + code_body + tests.
        
        # Let's try to match by string content.
        # This is heuristics-heavy.
        
        # Alternative: Just tokenize code_body and try to find it in full_code tokens?
        # Or simpler:
        # If compilation failed, punish everything? No, we want dense rewards.
        
        # Let's rely on the fact that we constructed full_code.
        # We can find where code_body starts in full_code.
        
        code_body_start_idx = full_code.find(code_body)
        if code_body_start_idx != -1:
            # We have character range of code_body in full_code.
            # We can map character positions to token indices in full_code tokenization (which is what code_token_rewards is aligned to).
            # create_token_rewards... returns a list of floats, one per token of the input code.
            # But wait, create_token_rewards... returns rewards for tokens of 'code'.
            
            # The 'code_token_rewards' list corresponds to tokens of 'full_code'.
            # We need to extract the slice corresponding to 'code_body'.
            
            # Use tokenizer offset mapping on full_code to find which tokens overlap with code_body range.
            full_encoding = processing_class(full_code, add_special_tokens=False, return_offsets_mapping=True)
            full_offsets = full_encoding['offset_mapping']
            
            # Find tokens within [code_body_start_idx, code_body_start_idx + len(code_body)]
            body_start = code_body_start_idx
            body_end = body_start + len(code_body)
            
            relevant_rewards = []
            for idx, (start, end) in enumerate(full_offsets):
                # If token is substantially within the body
                if start >= body_start and end <= body_end:
                    if idx < len(code_token_rewards):
                        relevant_rewards.append(code_token_rewards[idx])
            
            # Now map relevant_rewards to response tokens.
            # response might contain markdown wrappers around code_body.
            # We can try to align relevant_rewards to the code_body part of response.
            
            if relevant_rewards:
                 # Find code_body in response
                response_body_start = response.find(code_body)
                if response_body_start != -1:
                    # Align similarly
                    response_encoding = processing_class(response, add_special_tokens=False, return_offsets_mapping=True)
                    response_offsets = response_encoding['offset_mapping']
                    
                    r_idx = 0
                    for t_idx, (start, end) in enumerate(response_offsets):
                        if start >= response_body_start and end <= response_body_start + len(code_body):
                            if r_idx < len(relevant_rewards):
                                if t_idx < response_len:
                                    seq_rewards[t_idx] = relevant_rewards[r_idx]
                                r_idx += 1
        
        # If alignment fails, we might just default to clean_reward or last reward?
        # Or if compilation failed completely (success=False), maybe punish all tokens?
        # But we have `code_token_rewards` which should handle it.
        
        # Fallback: if we couldn't align, maybe just use the tail of code_token_rewards (heuristic)
        # But full_code has tests at the end, so tail is tests.
        # We need the middle part.
        
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
