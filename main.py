#!/usr/bin/env python3

import sys
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from src.models import CulinaryLanguageModel, TextProcessor, TextGenerator
from src.utils import CulinaryDataset, ModelTrainer, create_culinary_dataset, get_data_source_info
from src.utils.logger import logger
from src.chatbot import CulinaryChatbot
from src.config import ModelConfig, TrainingConfig, DataConfig, ChatbotConfig


def setup_environment() -> str:
    """Setup the environment and return the device."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    Path("models").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)

    return device


def show_data_info() -> None:
    """Show information about data sources."""
    logger.info("Analyzing data sources")
    
    data_info = get_data_source_info()
    
    logger.info("\n" + "="*50)
    logger.info("DATA SOURCES ANALYSIS")
    logger.info("="*50)
    
    if data_info["using_external_data"]:
        logger.info("Using external data files")
        stats = data_info["external_data_stats"]
        logger.info(f"Total files: {stats['total_files']}")
        logger.info(f"Total lines: {stats['total_lines']}")
        logger.info(f"File types: {stats['file_types']}")
        logger.info(f"File sizes: {stats['file_sizes']}")
    else:
        logger.warning("No external data files found")
    
    logger.info("\nFILE VALIDATION:")
    validation = data_info["file_validation"]
    for filename, status in validation.items():
        if status["exists"]:
            logger.info(f"{filename}: {status['lines']} lines")
        else:
            logger.warning(f"{filename}: Not found")
    
    print("="*50)


def train_model(device: str) -> None:
    """Train the culinary language model."""
    logger.info("Starting model training")

    model_config = ModelConfig()
    training_config = TrainingConfig()
    data_config = DataConfig()

    logger.info("Creating dataset")
    texts = create_culinary_dataset()
    logger.info(f"Dataset created with {len(texts)} texts")

    logger.info("Preparing text processor")
    processor = TextProcessor(max_sequence_length=model_config.max_sequence_length)
    processor.build_vocabulary(texts)
    logger.info(f"Vocabulary built with {processor.vocabulary_size} tokens")

    logger.info("Creating training sequences")
    sequences = processor.create_sequences(texts)
    logger.info(f"Created {len(sequences)} training sequences")

    dataset = CulinaryDataset(sequences, processor.max_sequence_length)
    dataloader = DataLoader(dataset, batch_size=training_config.batch_size, shuffle=True, num_workers=4)

    logger.info("Initializing model")
    model = CulinaryLanguageModel(
        vocabulary_size=processor.vocabulary_size,
        embedding_dim=model_config.embedding_dim,
        hidden_dim=model_config.hidden_dim,
        num_layers=model_config.num_layers,
        dropout=model_config.dropout,
    )

    model_info = model.get_model_info()
    logger.info(f"Model initialized with {model_info['total_parameters']:,} parameters")

    logger.info("Starting training")
    trainer = ModelTrainer(
        model=model,
        device=device,
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    trainer.train(
        train_dataloader=dataloader,
        epochs=training_config.epochs,
        save_path=training_config.model_save_path,
        early_stopping_patience=training_config.early_stopping_patience,
    )

    logger.info("Saving processor")
    processor.save(data_config.processor_save_path)

    logger.info("Training completed successfully")


def load_trained_model(device: str) -> tuple[CulinaryLanguageModel, TextProcessor]:
    """Load the trained model and processor."""
    logger.info("Loading trained model")

    processor = TextProcessor.load("models/processor.pkl")

    model = CulinaryLanguageModel(processor.vocabulary_size)
    checkpoint = torch.load("models/best_model.pth", map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    return model, processor


def demo_model(device: str) -> None:
    """Demonstrate the model's text generation capabilities."""
    logger.info("Starting model demonstration")

    try:
        model, processor = load_trained_model(device)
    except FileNotFoundError:
        logger.error("Model not found. Please train the model first with 'python main.py train'")
        return

    generator = TextGenerator(model, processor, device)

    test_phrases = ["To prepare", "My advice", "The cooking", "This dessert"]

    logger.info("Running generation tests")

    for phrase in test_phrases:
        generated = generator.generate(phrase, max_length=30)
        logger.info(f"Start: '{phrase}' -> Generation: '{generated}'")

    logger.info("Calculating generation statistics")
    stats = generator.get_generation_stats("To prepare", num_samples=5)
    for key, value in stats.items():
        logger.info(f"{key}: {value}")


def interactive_chat(device: str) -> None:
    """Start interactive chat interface."""
    logger.info("Starting interactive chat")

    try:
        model, processor = load_trained_model(device)
    except FileNotFoundError:
        logger.error("Model not found. Please train the model first with 'python main.py train'")
        return

    chatbot_config = ChatbotConfig()
    chatbot = CulinaryChatbot(model, processor, device, chatbot_config)
    chatbot.chat()


def run_tests(device: str) -> None:
    """Run automatic tests."""
    logger.info("Running automatic tests")

    try:
        model, processor = load_trained_model(device)
    except FileNotFoundError:
        logger.error("Model not found. Please train the model first with 'python main.py train'")
        return

    chatbot_config = ChatbotConfig()
    chatbot = CulinaryChatbot(model, processor, device, chatbot_config)

    test_questions = [
        "How to make a recipe?",
        "Cooking advice",
        "Tell me about saffron",
        "Dessert recipe",
        "Cooking technique",
    ]

    logger.info("Running intent detection and generation tests")

    for question in test_questions:
        intent_info = chatbot.get_intent_info(question)
        response = chatbot.generate_response(question)
        logger.info(f"Q: {question} | Intent: {intent_info['intent']} | Response: {response}")

    logger.info("Tests completed")


def show_help() -> None:
    """Display help information."""
    help_text = """
Culinary Assistant LLM - Help

Usage:
    python main.py [COMMAND]

Available commands:
    train   - Train the language model
    demo    - Quick model demonstration
    chat    - Interactive chat interface
    test    - Automatic tests
    data    - Show data sources information
    help    - Show this help

Examples:
    python main.py train    # Train the model
    python main.py demo     # See generation examples
    python main.py chat     # Chat with the assistant
    python main.py test     # Test functionalities
    python main.py data     # Show data sources info

Notes:
    - First time, use 'train' to train the model
    - Model will be saved in 'models/' folder
    - Use 'chat' for real-time interaction
    - Use 'data' to check external data files
    """
    logger.info(help_text)


def main() -> None:
    """Main entry point."""
    logger.info("Culinary Assistant LLM")
    logger.info("Development of an LLM with Deep Learning and NLP")
    logger.info("=" * 50)

    device = setup_environment()

    if len(sys.argv) < 2:
        show_help()
        return

    command = sys.argv[1].lower()

    try:
        if command == "train":
            train_model(device)
        elif command == "demo":
            demo_model(device)
        elif command == "chat":
            interactive_chat(device)
        elif command == "test":
            run_tests(device)
        elif command == "data":
            show_data_info()
        elif command == "help":
            show_help()
        else:
            logger.error(f"Unknown command: {command}")
            show_help()

    except KeyboardInterrupt:
        logger.info("User requested stop")
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.info("Use 'python main.py help' to see available commands")


if __name__ == "__main__":
    main()
