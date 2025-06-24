"""
Unit tests for the models
"""

import unittest
import torch
import sys
import os
from src.models import CulinaryLanguageModel, TextProcessor, TextGenerator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestTextProcessor(unittest.TestCase):
    """Tests for the TextProcessor class."""

    def setUp(self):
        """Setup for the tests."""
        self.processor = TextProcessor()
        self.test_texts = [
            "To prepare a delicious ratatouille",
            "Traditional veal blanquette",
            "The secret of a creamy risotto",
        ]

    def test_clean_text(self):
        """Test text cleaning."""
        dirty_text = "  To prepare a DELICIOUS ratatouille!  "
        cleaned = self.processor.clean_text(dirty_text)
        expected = "to prepare a delicious ratatouille"
        self.assertEqual(cleaned, expected)

    def test_build_vocabulary(self):
        """Test vocabulary building."""
        self.processor.build_vocabulary(self.test_texts)

        self.assertIn("<PAD>", self.processor.word_to_index)
        self.assertIn("<UNK>", self.processor.word_to_index)
        self.assertIn("<START>", self.processor.word_to_index)
        self.assertIn("<END>", self.processor.word_to_index)

        self.assertGreater(self.processor.vocabulary_size, 10)

    def test_text_to_indices(self):
        """Test text to indices conversion."""
        self.processor.build_vocabulary(self.test_texts)
        indices = self.processor.text_to_indices("to prepare")

        self.assertTrue(all(isinstance(idx, int) for idx in indices))
        self.assertGreater(len(indices), 0)

    def test_indices_to_text(self):
        """Test indices to text conversion."""
        self.processor.build_vocabulary(self.test_texts)
        original_text = "to prepare"
        indices = self.processor.text_to_indices(original_text)
        reconstructed_text = self.processor.indices_to_text(indices)

        self.assertEqual(reconstructed_text, original_text)

    def test_create_sequences(self):
        """Test sequence creation."""
        self.processor.build_vocabulary(self.test_texts)
        sequences = self.processor.create_sequences(self.test_texts)

        self.assertGreater(len(sequences), 0)

        for input_seq, target in sequences:
            self.assertIsInstance(input_seq, list)
            self.assertIsInstance(target, int)
            self.assertGreater(len(input_seq), 0)


class TestCulinaryLanguageModel(unittest.TestCase):
    """Tests for the CulinaryLanguageModel class."""

    def setUp(self):
        """Setup for the tests."""
        self.vocabulary_size = 100
        self.model = CulinaryLanguageModel(self.vocabulary_size)

    def test_model_initialization(self):
        """Test model initialization."""
        self.assertIsNotNone(self.model)
        self.assertEqual(self.model.vocabulary_size, self.vocabulary_size)

    def test_forward_pass(self):
        """Test forward pass."""
        batch_size = 2
        sequence_length = 10
        input_tensor = torch.randint(0, self.vocabulary_size, (batch_size, sequence_length))

        output = self.model(input_tensor)

        expected_shape = (batch_size, self.vocabulary_size)
        self.assertEqual(output.shape, expected_shape)

    def test_model_info(self):
        """Test model information retrieval."""
        info = self.model.get_model_info()

        required_keys = [
            "vocabulary_size",
            "embedding_dim",
            "hidden_dim",
            "num_layers",
            "total_parameters",
            "trainable_parameters",
        ]

        for key in required_keys:
            self.assertIn(key, info)
            self.assertIsNotNone(info[key])


class TestTextGenerator(unittest.TestCase):
    """Tests for the TextGenerator class."""

    def setUp(self):
        """Setup for the tests."""
        self.vocabulary_size = 50
        self.model = CulinaryLanguageModel(self.vocabulary_size)
        self.processor = TextProcessor()

        test_texts = ["to prepare", "the cooking", "this dessert"]
        self.processor.build_vocabulary(test_texts)

        self.generator = TextGenerator(self.model, self.processor, "cpu")

    def test_generator_initialization(self):
        """Test generator initialization."""
        self.assertIsNotNone(self.generator)
        self.assertEqual(self.generator.device, "cpu")

    def test_generate_text(self):
        """Test text generation."""
        seed_text = "to"
        generated = self.generator.generate(seed_text, max_length=5)

        self.assertIsInstance(generated, str)
        self.assertGreater(len(generated), len(seed_text))

    def test_generate_multiple(self):
        """Test multiple text generation."""
        seed_text = "to"
        samples = self.generator.generate_multiple(seed_text, num_samples=3)

        self.assertEqual(len(samples), 3)

        for sample in samples:
            self.assertIsInstance(sample, str)

    def test_generation_stats(self):
        """Test generation statistics."""
        seed_text = "to"
        stats = self.generator.get_generation_stats(seed_text, num_samples=3)

        required_keys = [
            "num_samples",
            "avg_length",
            "min_length",
            "max_length",
            "unique_words",
            "vocabulary_diversity",
        ]

        for key in required_keys:
            self.assertIn(key, stats)
            self.assertIsNotNone(stats[key])


if __name__ == "__main__":
    unittest.main()
