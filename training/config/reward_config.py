"""Unified reward configuration for DPO and PPO training.

Defines reward values and provides utilities for converting between
token-level and sequence-level rewards.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class RewardConfig:
    """
    Configuration for reward computation.

    Reward Curriculum (from RLSF paper):
    1. Syntactic Correctness: Learn to compile (avoid error_reward = 0.0)
    2. Runtime Safety: Learn not to crash (avoid runtime_error_reward = 0.2)
    3. Functional Correctness: Write clean code (aim for clean_reward = 1.0)
    """

    # Reward values
    error_reward: float = 0.0          # Compile errors (most severe penalty)
    runtime_error_reward: float = 0.2  # Runtime errors (better than compile errors)
    warning_reward: float = 0.5        # Warnings (minor issues)
    clean_reward: float = 1.0          # Clean code (best outcome)

    # Mode selection
    use_token_level: bool = True       # Token-level vs sequence-level rewards

    def sequence_reward_from_tokens(self, token_rewards: List[float]) -> float:
        """
        Compute sequence-level reward from token-level rewards.

        Uses mean pooling to aggregate token rewards into a single scalar.

        Args:
            token_rewards: List of per-token reward values

        Returns:
            Scalar reward (mean of token rewards)
        """
        if not token_rewards:
            return 0.0
        return sum(token_rewards) / len(token_rewards)

    def validate_token_rewards(self, token_rewards: List[float]) -> bool:
        """
        Validate that token rewards are within expected range.

        Args:
            token_rewards: List of per-token reward values

        Returns:
            True if all rewards are valid, False otherwise
        """
        if not token_rewards:
            return False

        valid_rewards = {
            self.error_reward,
            self.runtime_error_reward,
            self.warning_reward,
            self.clean_reward
        }

        for reward in token_rewards:
            if reward not in valid_rewards:
                return False

        return True
