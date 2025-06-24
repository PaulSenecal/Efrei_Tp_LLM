from .models import CulinaryLanguageModel, TextProcessor, TextGenerator
from .utils import CulinaryDataset, ModelTrainer, create_culinary_dataset
from .chatbot import CulinaryChatbot
from .config import ModelConfig, TrainingConfig, DataConfig, ChatbotConfig

__all__ = [
    "CulinaryLanguageModel",
    "TextProcessor",
    "TextGenerator",
    "CulinaryDataset",
    "ModelTrainer",
    "create_culinary_dataset",
    "CulinaryChatbot",
    "ModelConfig",
    "TrainingConfig",
    "DataConfig",
    "ChatbotConfig",
]

__version__ = "1.0.0"
__author__ = "Nolan, Aymene, Laurent, Paul, Benjamin"
__description__ = "Intelligent culinary assistant using LSTM LLM"
