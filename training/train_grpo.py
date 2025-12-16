import os
import sys
import torch
import json
from transformers import AutoTokenizer
from trl.trainer.grpo_config import GRPOConfig
from datasets import Dataset
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from grpo_trainer_token_rewards import TokenRewardGRPOTrainer
from cpp_pipeline.cpp_utils import compile_cpp_code, run_cpp_executable, create_token_rewards_from_compiler_errors, link_executable, clean_generated_code
from cpp_pipeline.load_data import get_multipl_e_dataset
from training.config.lora_config import get_unified_lora_config

# Global mapping to store tests for each prompt
PROMPT_TO_TESTS = {}

# Artifact logging for debugging reward computation
REWARD_ARTIFACTS_DIR = None
ARTIFACT_COUNTER = 0

def log_reward_artifact(
    prompt: str,
    raw_completion: str,
    cleaned_completion: str,
    code_body: str,
    tests: str,
    full_code: str,
    compile_success: bool,
    compiler_stderr: str,
    compiler_errors: list,
    runtime_success: bool = None,
    runtime_stderr: str = None,
    runtime_errors: list = None,
    final_reward: float = None,
):
    """
    Log detailed artifacts for debugging reward computation issues.
    Saves a JSON file with all intermediate values.
    """
    global ARTIFACT_COUNTER
    if REWARD_ARTIFACTS_DIR is None:
        return

    ARTIFACT_COUNTER += 1
    artifact = {
        "artifact_id": ARTIFACT_COUNTER,
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
        "raw_completion": raw_completion,
        "cleaned_completion": cleaned_completion,
        "code_body": code_body,
        "tests": tests[:500] + "..." if len(tests) > 500 else tests,  # Truncate long tests
        "full_code": full_code,
        "compile_success": compile_success,
        "compiler_stderr": compiler_stderr,
        "compiler_errors": compiler_errors,
        "runtime_success": runtime_success,
        "runtime_stderr": runtime_stderr,
        "runtime_errors": runtime_errors,
        "final_reward": final_reward,
    }

    artifact_path = os.path.join(REWARD_ARTIFACTS_DIR, f"reward_artifact_{ARTIFACT_COUNTER:05d}.json")
    try:
        with open(artifact_path, "w") as f:
            json.dump(artifact, f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to write artifact: {e}")

class LoggingTokenRewardGRPOTrainer(TokenRewardGRPOTrainer):
    """
    A subclass of TokenRewardGRPOTrainer that adds file-based logging for rewards and rollouts.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_log_path = os.path.join(self.args.output_dir, "grpo_rewards.jsonl")
        self.rollout_log_path = os.path.join(self.args.output_dir, "grpo_rollouts.jsonl")
        # Clear files at the beginning of a run on the main process
        if self.accelerator.is_main_process:
            # Create directory if it doesn't exist
            os.makedirs(self.args.output_dir, exist_ok=True)
            if os.path.exists(self.reward_log_path):
                os.remove(self.reward_log_path)
            if os.path.exists(self.rollout_log_path):
                os.remove(self.rollout_log_path)

    def log(self, metrics, start_time=None):
        """
        Overrides the default log method to write reward-related metrics and rollouts to JSONL files.
        """
        # Call original log method (for tensorboard etc.)
        super().log(metrics, start_time)

        # Also write to our custom files on the main process
        if self.accelerator.is_main_process:
            # Log scalar metrics
            log_metrics = {k: v for k, v in metrics.items() if "reward" in k or "advantage" in k}
            log_metrics["step"] = self.state.global_step
            try:
                with open(self.reward_log_path, "a") as f:
                    f.write(json.dumps(log_metrics) + "\n")
            except Exception as e:
                print(f"Warning: Failed to write to reward log file: {e}")
            
            # Log rollouts (completions, prompts, rewards, advantages)
            try:
                if hasattr(self, "_logs") and self._logs:
                    prompts = self._logs.get("prompt", [])
                    completions = self._logs.get("completion", [])
                    rewards_dict = self._logs.get("rewards", {})
                    advantages = self._logs.get("advantages", [])
                    
                    # Only log if we have data
                    if prompts and completions:
                        # Write each rollout as a separate JSONL entry
                        with open(self.rollout_log_path, "a") as f:
                            num_rollouts = len(prompts)
                            for i in range(num_rollouts):
                                # Get advantage value (could be scalar or token-level list)
                                adv_value = advantages[i] if i < len(advantages) else None
                                
                                rollout_entry = {
                                    "step": self.state.global_step,
                                    "prompt": prompts[i] if i < len(prompts) else "",
                                    "completion": completions[i] if i < len(completions) else "",
                                    "rewards": {name: rewards_dict[name][i] if name in rewards_dict and i < len(rewards_dict[name]) else None 
                                               for name in rewards_dict},
                                }
                                
                                # Handle advantages: could be scalar or token-level (nested list)
                                if adv_value is not None:
                                    # Flatten if it's a nested list (token-level advantages)
                                    if isinstance(adv_value, (list, tuple)) and len(adv_value) > 0:
                                        if isinstance(adv_value[0], (list, tuple)):
                                            # Nested list - flatten it
                                            adv_flat = [item for sublist in adv_value for item in (sublist if isinstance(sublist, (list, tuple)) else [sublist])]
                                        else:
                                            adv_flat = list(adv_value)
                                        
                                        # Store full token-level advantages and summary
                                        rollout_entry["advantage_tokens"] = [float(x) for x in adv_flat]
                                        rollout_entry["advantage_summary"] = {
                                            "mean": float(sum(adv_flat) / len(adv_flat)) if adv_flat else 0.0,
                                            "min": float(min(adv_flat)) if adv_flat else 0.0,
                                            "max": float(max(adv_flat)) if adv_flat else 0.0,
                                            "length": len(adv_flat),
                                        }
                                        # Also store scalar summary for convenience
                                        rollout_entry["advantage"] = rollout_entry["advantage_summary"]["mean"]
                                    else:
                                        # Scalar advantage
                                        rollout_entry["advantage"] = float(adv_value) if isinstance(adv_value, (int, float)) else adv_value
                                else:
                                    rollout_entry["advantage"] = None
                                
                                f.write(json.dumps(rollout_entry) + "\n")
            except Exception as e:
                print(f"Warning: Failed to write to rollout log file: {e}")
                import traceback
                traceback.print_exc()

def get_prompts():
    """Get prompts from MultiPL-E dataset."""
    examples = get_multipl_e_dataset(split="train")
    
    # Populate the global mapping
    global PROMPT_TO_TESTS
    PROMPT_TO_TESTS = {}
    for ex in examples:
        PROMPT_TO_TESTS[ex["prompt"]] = ex["tests"]
        
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

        # Retrieve tests
        tests = PROMPT_TO_TESTS.get(prompt, "")

        # Clean completion - removes int main() and everything after
        cleaned_response = clean_generated_code(text)

        # Simple extraction heuristic - if markdown is used, trust it, otherwise use cleaned completion
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
            # Count braces to see if the function is closed
            # Simple heuristic: if there's a closing brace, the model closed the function
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

        # Initialize runtime variables for artifact logging
        runtime_success = None
        runtime_stderr = None
        runtime_errors = None

        # Compute scalar reward based on compilation and runtime
        if not success:
            # Compilation failed
            reward = 0.0
        else:
            # Compilation succeeded, try to run
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
                f.write(full_code)
                src_path = f.name

            exe_path = src_path + ".exe"
            runtime_success = False
            if link_executable(src_path, exe_path):
                run_success, _, run_stderr, run_errors = run_cpp_executable(exe_path)
                runtime_success = run_success
                runtime_stderr = run_stderr
                runtime_errors = run_errors

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

        # Log artifact for debugging
        log_reward_artifact(
            prompt=prompt,
            raw_completion=text,
            cleaned_completion=cleaned_response,
            code_body=code_body,
            tests=tests,
            full_code=full_code,
            compile_success=success,
            compiler_stderr=stderr,
            compiler_errors=errors,
            runtime_success=runtime_success,
            runtime_stderr=runtime_stderr,
            runtime_errors=runtime_errors,
            final_reward=reward,
        )

        rewards.append(reward)

    return rewards

def main():
    global REWARD_ARTIFACTS_DIR

    model_name = "Qwen/Qwen2.5-Coder-0.5B"
    output_dir = os.environ.get("TRAINING_OUTPUT_DIR", "grpo_results")
    use_token_level_rewards_str = os.environ.get("USE_TOKEN_LEVEL_REWARDS", "True")
    use_token_level_rewards = use_token_level_rewards_str.lower() in ("true", "1", "t")

    max_rows_str = os.environ.get("MAX_ROWS")
    max_rows = int(max_rows_str) if max_rows_str else None

    # Set up artifact logging directory
    REWARD_ARTIFACTS_DIR = os.path.join(output_dir, "reward_artifacts")
    os.makedirs(REWARD_ARTIFACTS_DIR, exist_ok=True)
    print(f"Logging reward artifacts to: {REWARD_ARTIFACTS_DIR}")

    print(f"Starting GRPO training...")
    print(f"Token-level rewards enabled: {use_token_level_rewards}")
    if max_rows:
        print(f"Limiting dataset to {max_rows} rows")

    # 1. Load Data (Prompts)
    prompts = get_prompts()
    original_size = len(prompts)
    if max_rows:
        prompts = prompts[:max_rows]
        print(f"Limited dataset from {original_size} to {len(prompts)} prompts")
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

            # Decode prompts and completions
            completions = processing_class.batch_decode(completion_ids, skip_special_tokens=True)
            prompts = processing_class.batch_decode(prompt_ids, skip_special_tokens=True)

            for i in range(batch_size):
                text = completions[i]
                prompt = prompts[i]
                
                # Retrieve tests
                tests = PROMPT_TO_TESTS.get(prompt, "")
                
                # Clean completion - removes int main() and everything after
                cleaned_response = clean_generated_code(text)

                # Simple extraction heuristic
                if "```cpp" in cleaned_response:
                    code_body = cleaned_response.split("```cpp")[1].split("```")[0]
                elif "```c++" in cleaned_response:
                    code_body = cleaned_response.split("```c++")[1].split("```")[0]
                elif "```" in cleaned_response:
                    code_body = cleaned_response.split("```")[1].split("```")[0]
                else:
                    code_body = cleaned_response

                # CRITICAL FIX: Strip trailing } if model already closed the function
                # (Same fix as in scalar reward function)
                code_body_stripped = code_body.rstrip()
                if code_body_stripped.endswith('}'):
                    open_count = code_body_stripped.count('{')
                    close_count = code_body_stripped.count('}')
                    if close_count > open_count:
                        last_brace = code_body_stripped.rfind('}')
                        code_body = code_body_stripped[:last_brace]

                # Stitch full code
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
                # Strategy: Compute rewards for code_body, then align with completion tokens
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

                # Map code_body rewards to completion tokens
                completion_len = completion_ids.shape[1]
                seq_rewards = [1.0] * completion_len  # Default: clean_reward

                # Find code_body in text (completion might have markdown or extra whitespace)
                response_body_start = text.find(code_body)

                if response_body_start != -1 and len(code_body_rewards) > 0:
                    # Tokenize completion (text) to get offset mapping
                    response_encoding = processing_class(text, add_special_tokens=False, return_offsets_mapping=True)
                    response_offsets = response_encoding['offset_mapping']

                    # Tokenize code_body to get offset mapping
                    body_encoding = processing_class(code_body, add_special_tokens=False, return_offsets_mapping=True)
                    body_offsets = body_encoding['offset_mapping']

                    # Ensure lengths match
                    if len(code_body_rewards) != len(body_offsets):
                        print(f"WARNING: Reward length mismatch: {len(code_body_rewards)} rewards vs {len(body_offsets)} tokens")
                        code_body_rewards = code_body_rewards[:len(body_offsets)]  # Truncate if needed

                    # Map code_body tokens to completion tokens by character position
                    body_end = response_body_start + len(code_body)

                    reward_idx = 0
                    for resp_idx, (start, end) in enumerate(response_offsets):
                        # Check if this completion token overlaps with code_body region
                        if start >= response_body_start and end <= body_end and resp_idx < completion_len:
                            # Find corresponding reward
                            if reward_idx < len(code_body_rewards):
                                seq_rewards[resp_idx] = code_body_rewards[reward_idx]
                                reward_idx += 1
                else:
                    # Alignment failed - use fallback strategy
                    # If compilation or runtime failed, apply penalty to all tokens
                    if not success:
                        # Compile error - severe penalty
                        seq_rewards = [0.0] * completion_len
                        print(f"WARNING: Compilation failed, applying 0.0 penalty to all tokens")
                    elif runtime_errors:
                        # Runtime error - moderate penalty
                        seq_rewards = [0.2] * completion_len
                        print(f"WARNING: Runtime error, applying 0.2 penalty to all tokens")
                    else:
                        # Clean code - full reward
                        seq_rewards = [1.0] * completion_len

                # Logging for debugging
                non_default_rewards = [r for r in seq_rewards if r != 1.0]
                if non_default_rewards:
                    print(f"  Completion {i}: {len(non_default_rewards)}/{len(seq_rewards)} tokens have non-default rewards")
                    print(f"    Compile: {success}, Runtime errors: {len(runtime_errors)}, Adjusted errors in body: {len(adjusted_errors)}")

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
        max_completion_length=512,  # Increased to match inference and handle longer function bodies
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
