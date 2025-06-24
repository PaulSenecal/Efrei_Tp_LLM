"""
Parameters for the culinary assistant
"""

from dataclasses import dataclass, field
from typing import Optional, Dict


@dataclass
class ModelConfig:
    """Configuration of the language model."""

    embedding_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.3

    max_sequence_length: int = 50
    temperature: float = 0.8
    top_k: Optional[int] = None

    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"
    start_token: str = "<START>"
    end_token: str = "<END>"


@dataclass
class TrainingConfig:
    """Configuration of the training."""

    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    batch_size: int = 32
    epochs: int = 50

    early_stopping_patience: int = 50
    min_delta: float = 1e-4

    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 5

    max_grad_norm: float = 1.0

    save_best_model: bool = True
    model_save_path: str = "models/best_model.pth"
    checkpoint_save_path: str = "models/checkpoint.pth"


@dataclass
class DataConfig:
    """Configuration of the data."""

    train_data_path: Optional[str] = "src/data/recipes.txt"
    val_data_path: Optional[str] = "src/data/cooking_tips.csv"
    ingredients_data_path: Optional[str] = "src/data/ingredients.json"
    processor_save_path: str = "models/processor.pkl"

    val_split: float = 0.2
    random_seed: int = 42

    enable_data_augmentation: bool = True
    max_augmentation_factor: int = 3


@dataclass
class ChatbotConfig:
    """Configuration of the chatbot."""

    intent_keywords: Dict[str, list] = field(
        default_factory=lambda: {
            "recipe": ["recipe", "prepare", "cook", "make"],
            "advice": ["advice", "tip", "how", "technique"],
            "ingredient": ["ingredient", "product", "food"],
            "cooking": ["cooking", "cook", "temperature", "time"],
            "dessert": ["dessert", "cake", "pie", "sweet"],
            "dish": ["dish", "meat", "fish", "vegetable"],
        }
    )

    intent_starters: Dict[str, str] = field(
        default_factory=lambda: {
            "recipe": "To prepare this dish",
            "advice": "My culinary advice",
            "ingredient": "This ingredient",
            "cooking": "For cooking",
            "dessert": "This delicious dessert",
            "dish": "This savory dish",
            "general": "In cooking",
        }
    )

    max_response_length: int = 40
    default_temperature: float = 0.8


DEFAULT_CONFIG = {
    "model": ModelConfig(),
    "training": TrainingConfig(),
    "data": DataConfig(),
    "chatbot": ChatbotConfig(),
}
