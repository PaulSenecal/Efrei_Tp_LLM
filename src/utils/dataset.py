import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict


class CulinaryDataset(Dataset):
    """Dataset for culinary text sequences."""

    def __init__(self, sequences: List[Tuple[List[int], int]], max_length: int):
        """Initialize the dataset."""
        self.sequences = sequences
        self.max_length = max_length

    def __len__(self) -> int:
        """Return the number of sequences."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sequence and its target."""
        input_seq, target = self.sequences[idx]

        padding_length = self.max_length - len(input_seq)
        padded_input = [0] * padding_length + input_seq

        return torch.tensor(padded_input, dtype=torch.long), torch.tensor(target, dtype=torch.long)

    def get_dataset_info(self) -> Dict:
        """Get dataset information."""
        sequence_lengths = [len(seq) for seq, _ in self.sequences]

        return {
            "num_samples": len(self.sequences),
            "max_sequence_length": self.max_length,
            "avg_sequence_length": sum(sequence_lengths) / len(sequence_lengths),
            "min_sequence_length": min(sequence_lengths),
            "max_actual_length": max(sequence_lengths),
        }
