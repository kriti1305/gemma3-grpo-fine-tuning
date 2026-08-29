import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducible experiments.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    """
    Detect the available compute device.
    """
    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


def print_environment() -> None:
    """
    Display basic information about the training environment.
    """
    device = get_device()

    print("=" * 50)
    print("Training Environment")
    print("=" * 50)
    print(f"PyTorch version : {torch.__version__}")
    print(f"Device          : {device}")

    if device == "cuda":
        print(f"GPU             : {torch.cuda.get_device_name(0)}")
    else:
        print("GPU             : Not available")

    print("=" * 50)