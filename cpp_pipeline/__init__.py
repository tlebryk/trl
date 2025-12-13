"""
C++ Code Generation Pipeline for RLSF

This package provides a modular pipeline for:
1. Creating C++ code examples
2. Compiling code and gathering compiler feedback
3. Running executables and gathering runtime feedback
4. Computing token-level rewards from feedback
5. Preparing datasets for training
"""

from .create_examples import create_all_examples
from .compile_examples import compile_all_examples
from .run_examples import run_all_examples
from .compute_rewards import compute_rewards_for_all_examples
from .prepare_dataset import prepare_dataset

__all__ = [
    "create_all_examples",
    "compile_all_examples",
    "run_all_examples",
    "compute_rewards_for_all_examples",
    "prepare_dataset",
]
