"""
Main chatbot for the culinary assistant
"""

from typing import Dict, List
from .models import CulinaryLanguageModel, TextProcessor, TextGenerator
from .config import ChatbotConfig
from .utils.logger import logger


class CulinaryChatbot:
    """
    Intelligent chatbot specialized in the culinary domain.

    Features:
    - Intent detection based on keywords
    - Contextual response generation
    - Interactive conversational interface
    """

    def __init__(
        self,
        model: CulinaryLanguageModel,
        processor: TextProcessor,
        device: str = "cpu",
        config: ChatbotConfig = None,
    ):
        """
        Initialize the culinary chatbot.

        Args:
            model: Trained language model
            processor: Text processor
            device: Calcul device
            config: Chatbot configuration
        """
        self.generator = TextGenerator(model, processor, device)
        self.processor = processor
        self.device = device
        self.config = config or ChatbotConfig()

        self.intent_keywords = self.config.intent_keywords
        self.intent_starters = self.config.intent_starters

    def detect_intent(self, user_input: str) -> str:
        """
        Detect the user's intent.

        Args:
            user_input: User input

        Returns:
            Detected intent
        """
        user_input_lower = user_input.lower()

        for intent, keywords in self.intent_keywords.items():
            if any(keyword in user_input_lower for keyword in keywords):
                return intent

        return "general"

    def generate_response(self, user_input: str, temperature: float = None) -> str:
        """
        Generate a response based on the user's input.

        Args:
            user_input: User input
            temperature: Temperature for generation

        Returns:
            Generated response
        """
        intent = self.detect_intent(user_input)
        starter = self.intent_starters[intent]

        temp = temperature or self.config.default_temperature
        response = self.generator.generate(
            starter, max_length=self.config.max_response_length, temperature=temp
        )

        return self.format_response(response)

    def format_response(self, response: str) -> str:
        """
        Format the response for display.

        Args:
            response: Raw response

        Returns:
            Formatted response
        """
        response = response.strip()

        if not response.endswith((".", "!", "?")):
            response += "."

        if response:
            response = response[0].upper() + response[1:]

        return response

    def get_intent_info(self, user_input: str) -> Dict:
        """
        Return detailed information about the detected intent.

        Args:
            user_input: User input

        Returns:
            Dictionary with intent information
        """
        intent = self.detect_intent(user_input)
        keywords_found = []

        user_input_lower = user_input.lower()
        if intent in self.intent_keywords:
            for keyword in self.intent_keywords[intent]:
                if keyword in user_input_lower:
                    keywords_found.append(keyword)
            if self.intent_keywords[intent]:
                confidence = len(keywords_found) / len(self.intent_keywords[intent])
            else:
                confidence = 0.0
        else:
            confidence = 0.0

        return {
            "intent": intent,
            "confidence": confidence,
            "keywords_found": keywords_found,
            "starter_phrase": self.intent_starters[intent],
        }

    def chat(self, max_turns: int = 50) -> None:
        """
        Launch the interactive chat interface.

        Args:
            max_turns: Maximum number of conversation turns
        """
        self._print_welcome_message()

        turn_count = 0

        while turn_count < max_turns:
            try:
                user_input = input("You: ").strip()
                turn_count += 1

                if user_input.lower() in ["quit", "exit", "q"]:
                    print("Culinary Assistant: Goodbye and bon appétit!")
                    break

                if user_input.lower() == "help":
                    self._show_help()
                    continue

                if user_input.lower() == "intent":
                    self._show_intent_info()
                    continue

                if user_input.lower() == "stats":
                    self._show_model_stats()
                    continue

                if not user_input:
                    print("Culinary Assistant: I didn't hear your question. Can you repeat?")
                    continue

                response = self.generate_response(user_input)
                print(f"Culinary Assistant: {response}\n")

            except KeyboardInterrupt:
                print("\nCulinary Assistant: Conversation interrupted. Goodbye!")
                break
            except Exception as e:
                logger.error(f"Chat error: {e}")
                print("Culinary Assistant: Sorry, I encountered a problem. Can you rephrase?")

    def _print_welcome_message(self) -> None:
        """Print the welcome message."""
        print("Culinary Assistant: Hello! I'm your virtual culinary assistant.")
        print("Ask me for recipes, tips or cooking information.")
        print("Special commands:")
        print("  - 'quit' or 'exit': End conversation")
        print("  - 'help': Show help")
        print("  - 'intent': See detected intent")
        print("  - 'stats': Model statistics")
        print("-" * 60)

    def _show_help(self) -> None:
        """Display the chatbot's help."""
        print("\nHELP - Culinary Assistant")
        print("=" * 40)
        print("You can ask me for:")
        print("• Recipes and cooking techniques")
        print("• Culinary tips")
        print("• Ingredient information")
        print("• Cooking techniques")
        print("• Dessert recipes")
        print("• Main dishes")
        print("\nExamples:")
        print("• 'How to make a recipe?'")
        print("• 'Cooking tip'")
        print("• 'Tell me about saffron'")
        print("• 'Dessert recipe'")
        print("-" * 40)

    def _show_intent_info(self) -> None:
        """Display the intent information."""
        print("\nDETECTED INTENTS")
        print("=" * 30)
        for intent, keywords in self.intent_keywords.items():
            print(f"{intent}: {', '.join(keywords)}")
        print("-" * 30)

    def _show_model_stats(self) -> None:
        """Display the model statistics."""
        print("\nMODEL STATISTICS")
        print("=" * 30)

        vocab_info = self.processor.get_vocabulary_info()
        print(f"Vocabulary size: {vocab_info['vocabulary_size']}")
        print(f"Max sequence length: {vocab_info['max_sequence_length']}")
        print(f"Device: {self.device}")

        model_info = self.generator.model.get_model_info()
        print(f"• Total parameters: {model_info['total_parameters']:,}")
        print(f"• Trainable parameters: {model_info['trainable_parameters']:,}")
        print("-" * 30)

    def batch_generate(self, questions: List[str], temperature: float = None) -> List[str]:
        """
        Generate responses for a list of questions.

        Args:
            questions: List of questions
            temperature: Temperature for generation

        Returns:
            List of responses
        """
        responses = []
        for question in questions:
            response = self.generate_response(question, temperature)
            responses.append(response)

        return responses
