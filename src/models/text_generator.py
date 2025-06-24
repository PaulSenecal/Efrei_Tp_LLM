import torch
import torch.nn.functional as F
from typing import List, Optional, Dict
from .language_model import CulinaryLanguageModel
from .text_processor import TextProcessor


class TextGenerator:
    """Text generator using the trained language model."""

    def __init__(self, model: CulinaryLanguageModel, processor: TextProcessor, device: str = "cpu"):
        """Initialize the text generator."""
        self.model = model.to(device)
        self.processor = processor
        self.device = device

    def generate(
        self,
        seed_text: str,
        max_length: int = 50,
        temperature: float = 0.8,
        top_k: Optional[int] = None,
    ) -> str:
        """Generate text from a seed text."""
        self.model.eval()

        current_indices = [
            self.processor.word_to_index[self.processor.special_tokens["START"]]
        ] + self.processor.text_to_indices(seed_text)
        generated_text = seed_text

        with torch.no_grad():
            for _ in range(max_length):
                input_indices = self._prepare_input_sequence(current_indices)
                input_tensor = torch.tensor([input_indices], dtype=torch.long).to(self.device)
                output = self.model(input_tensor)

                logits = output[0] / temperature

                if top_k is not None:
                    logits = self._apply_top_k_sampling(logits, top_k)

                probabilities = F.softmax(logits, dim=0)
                next_index = torch.multinomial(probabilities, 1).item()

                if next_index == self.processor.word_to_index[self.processor.special_tokens["END"]]:
                    break

                next_word = self.processor.index_to_word.get(
                    next_index, self.processor.special_tokens["UNK"]
                )
                generated_text += " " + next_word
                current_indices.append(next_index)

                if next_word in [".", "!", "?", ";", ":"]:
                    break

        return generated_text

    def _prepare_input_sequence(self, current_indices: List[int]) -> List[int]:
        """Prepare input sequence with proper padding."""
        if len(current_indices) > self.processor.max_sequence_length:
            return current_indices[-self.processor.max_sequence_length :]

        padding_length = self.processor.max_sequence_length - len(current_indices)
        return [0] * padding_length + current_indices

    def _apply_top_k_sampling(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Apply top-k sampling to logits."""
        top_k_logits, top_k_indices = torch.topk(logits, top_k)
        filtered_logits = torch.full_like(logits, float("-inf"))
        filtered_logits.scatter_(0, top_k_indices, top_k_logits)
        return filtered_logits

    def generate_multiple(
        self, seed_text: str, num_samples: int = 3, max_length: int = 50, temperature: float = 0.8
    ) -> List[str]:
        """Generate multiple text samples."""
        samples = []
        for _ in range(num_samples):
            sample = self.generate(seed_text, max_length, temperature)
            samples.append(sample)

        return samples

    def generate_with_beam_search(
        self, seed_text: str, max_length: int = 50, beam_width: int = 5
    ) -> str:
        """Generate text using beam search."""
        self.model.eval()

        current_indices = [
            self.processor.word_to_index[self.processor.special_tokens["START"]]
        ] + self.processor.text_to_indices(seed_text)

        beam_states = [(current_indices, 0.0)]

        with torch.no_grad():
            for _ in range(max_length):
                new_beam_states = []

                for sequence, score in beam_states:
                    input_indices = self._prepare_input_sequence(sequence)
                    input_tensor = torch.tensor([input_indices], dtype=torch.long).to(self.device)
                    output = self.model(input_tensor)
                    logits = output[0]

                    top_k_logits, top_k_indices = torch.topk(logits, beam_width)

                    for logit, idx in zip(top_k_logits, top_k_indices):
                        if (
                            idx.item()
                            == self.processor.word_to_index[self.processor.special_tokens["END"]]
                        ):
                            new_beam_states.append((sequence, score + logit.item()))
                        else:
                            new_sequence = sequence + [idx.item()]
                            new_score = score + logit.item()
                            new_beam_states.append((new_sequence, new_score))

                beam_states = sorted(new_beam_states, key=lambda x: x[1], reverse=True)[:beam_width]

                if all(
                    seq[-1] == self.processor.word_to_index[self.processor.special_tokens["END"]]
                    for seq, _ in beam_states
                ):
                    break

        best_sequence, _ = beam_states[0]
        return self.processor.indices_to_text(best_sequence)

    def get_generation_stats(self, seed_text: str, num_samples: int = 10) -> Dict:
        """Get statistics about text generation."""
        samples = self.generate_multiple(seed_text, num_samples)

        lengths = [len(sample.split()) for sample in samples]
        unique_words = set()
        for sample in samples:
            unique_words.update(sample.split())

        return {
            "num_samples": num_samples,
            "avg_length": sum(lengths) / len(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "unique_words": len(unique_words),
            "vocabulary_diversity": len(unique_words) / sum(lengths),
        }
