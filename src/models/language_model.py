import torch
import torch.nn as nn
from typing import Dict


class CulinaryLanguageModel(nn.Module):
    """LSTM-based language model for culinary text generation."""

    def __init__(
        self,
        vocabulary_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        """Initialize the language model."""
        super(CulinaryLanguageModel, self).__init__()

        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocabulary_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocabulary_size)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        last_hidden = lstm_out[:, -1, :]
        dropped = self.dropout(last_hidden)
        fc1_out = self.relu(self.fc1(dropped))
        output = self.fc2(fc1_out)
        return output

    def get_model_info(self) -> Dict:
        """Get model information and statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "vocabulary_size": self.vocabulary_size,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        }
