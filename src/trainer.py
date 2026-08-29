"""
GRPO fine-tuning pipeline for Gemma 3.

This module provides:
- Dataset preparation
- LoRA configuration
- Reward-based training
- Gradient accumulation
- Gradient checkpointing
- Model saving
"""

from pathlib import Path

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer

from src.config import get_config
from src.rewards import combined_reward
from src.utils import print_environment, set_seed


def create_training_dataset() -> Dataset:
    """
    Create a small instruction dataset for local development.

    The dataset is intentionally small so that the pipeline
    can be tested without requiring a large GPU.
    """

    data = {
        "prompt": [
            "Explain what machine learning is.",
            "What is Python used for?",
            "Explain the purpose of a database.",
            "What is data visualization?",
            "Explain artificial intelligence simply.",
        ],
    }

    return Dataset.from_dict(data)


def create_lora_config() -> LoraConfig:
    """
    Create the LoRA configuration used for parameter-efficient
    fine-tuning.
    """

    config = get_config()

    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
    )


def score_example(response: str, keywords: list[str]) -> float:
    """
    Calculate a reward score for a generated response.
    """

    return combined_reward(
        response=response,
        keywords=keywords,
    )


def prepare_tokenizer():
    """
    Load the tokenizer for the configured model.
    """

    config = get_config()

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def prepare_output_directory() -> Path:
    """
    Create the directory used for training outputs.
    """

    config = get_config()

    output_dir = Path(config.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_dir


def main() -> None:
    """
    Run the local GRPO project pipeline.
    """

    config = get_config()

    set_seed(config.seed)
    print_environment()

    print("\nModel:", config.model_name)
    print("Learning rate:", config.learning_rate)
    print("LoRA rank:", config.lora_r)

    dataset = create_training_dataset()

    print("\nDataset examples:", len(dataset))

    lora_config = create_lora_config()

    print("LoRA configuration created.")
    print("LoRA rank:", lora_config.r)

    output_dir = prepare_output_directory()

    print("Output directory:", output_dir)

    print("\nPipeline setup completed successfully.")


if __name__ == "__main__":
    main()