import re
import pickle
from typing import List, Dict, Tuple


class TextProcessor:
    """Text processor for vocabulary management and sequence creation."""

    def __init__(self, max_sequence_length: int = 50):
        """Initialize the text processor."""
        self.word_to_index: Dict[str, int] = {}
        self.index_to_word: Dict[int, str] = {}
        self.vocabulary_size: int = 0
        self.max_sequence_length = max_sequence_length

        self.special_tokens = {"PAD": "<PAD>", "UNK": "<UNK>", "START": "<START>", "END": "<END>"}

    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        text = text.lower()
        text = re.sub(r"[^\w\s\-àâäéèêëïîôùûüÿæœç]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def build_vocabulary(self, texts: List[str]) -> None:
        """Build vocabulary from texts."""
        all_words = []

        for text in texts:
            words = self.clean_text(text).split()
            all_words.extend(words)

        unique_words = sorted(set(all_words))

        self.word_to_index[self.special_tokens["PAD"]] = 0
        self.word_to_index[self.special_tokens["UNK"]] = 1
        self.word_to_index[self.special_tokens["START"]] = 2
        self.word_to_index[self.special_tokens["END"]] = 3

        for idx, word in enumerate(unique_words):
            self.word_to_index[word] = idx + 4

        self.index_to_word = {idx: word for word, idx in self.word_to_index.items()}
        self.vocabulary_size = len(self.word_to_index)

    def text_to_indices(self, text: str) -> List[int]:
        """Convert text to indices."""
        words = self.clean_text(text).split()
        return [
            self.word_to_index.get(word, self.word_to_index[self.special_tokens["UNK"]])
            for word in words
        ]

    def indices_to_text(self, indices: List[int]) -> str:
        """Convert indices back to text."""
        words = []
        for idx in indices:
            if idx == 0:
                continue
            if idx in [
                self.word_to_index[self.special_tokens["START"]],
                self.word_to_index[self.special_tokens["END"]],
            ]:
                continue
            words.append(self.index_to_word.get(idx, self.special_tokens["UNK"]))

        return " ".join(words)

    def create_sequences(self, texts: List[str]) -> List[Tuple[List[int], int]]:
        """Create training sequences from texts."""
        sequences = []

        for text in texts:
            indices = (
                [self.word_to_index[self.special_tokens["START"]]]
                + self.text_to_indices(text)
                + [self.word_to_index[self.special_tokens["END"]]]
            )

            for i in range(1, len(indices)):
                input_seq = indices[:i]
                target = indices[i]

                if len(input_seq) <= self.max_sequence_length:
                    sequences.append((input_seq, target))

        return sequences

    def save(self, filepath: str) -> None:
        """Save the processor to file."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> "TextProcessor":
        """Load the processor from file."""
        with open(filepath, "rb") as f:
            return pickle.load(f)

    def get_vocabulary_info(self) -> Dict:
        """Get vocabulary information."""
        return {
            "vocabulary_size": self.vocabulary_size,
            "max_sequence_length": self.max_sequence_length,
            "special_tokens": list(self.special_tokens.values()),
            "most_common_words": list(self.word_to_index.keys())[:10],
        }
