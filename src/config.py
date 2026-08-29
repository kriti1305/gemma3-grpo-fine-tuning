from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Configuration for Gemma 3 GRPO fine-tuning."""

    model_name: str = "google/gemma-3-1b-it"

    output_dir: str = "./outputs"

    learning_rate: float = 1e-5
    num_train_epochs: int = 1

    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4

    max_prompt_length: int = 256
    max_completion_length: int = 256

    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    gradient_checkpointing: bool = True
    mixed_precision: bool = True

    seed: int = 42


def get_config() -> TrainingConfig:
    """Return the default training configuration."""
    return TrainingConfig()