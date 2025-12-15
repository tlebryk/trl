"""Unified LoRA configuration for DPO and PPO training.

CRITICAL: Both trainers must use identical target_modules to ensure
adapter compatibility for evaluation and comparison.
"""

from peft import LoraConfig, TaskType


def get_unified_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
) -> LoraConfig:
    """
    Get unified LoRA configuration used by both DPO and PPO trainers.

    This ensures that both training methods produce compatible LoRA adapters
    with the same architecture (same rank, same target modules), allowing for
    fair comparison during evaluation.

    Args:
        r: LoRA rank (default: 16)
        lora_alpha: LoRA scaling parameter (default: 32)
        lora_dropout: Dropout probability for LoRA layers (default: 0.05)

    Returns:
        LoraConfig configured for Qwen2.5-Coder model architecture
    """
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            # Attention projection matrices
            "q_proj",  # Query projection
            "k_proj",  # Key projection
            "v_proj",  # Value projection
            "o_proj",  # Output projection
            # MLP (Feed-forward) matrices
            "gate_proj",  # Gating projection
            "up_proj",    # Up projection
            "down_proj",  # Down projection
        ],
    )
