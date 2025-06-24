"""
Model trainer for the culinary assistant
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
import time

from .logger import logger


class ModelTrainer:
    """
    Model trainer for the culinary language model.

    Handles training, validation, and model saving.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
    ):
        """
        Initialize the trainer.

        Args:
            model: Model to train
            device: Calcul device (cpu/cuda)
            learning_rate: Learning rate
            weight_decay: L2 regularization
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.learning_rates: List[float] = []

    def train_epoch(self, dataloader: DataLoader) -> float:
        """
        Train the model for one epoch.

        Args:
            dataloader: DataLoader for training data

        Returns:
            Average loss over the epoch
        """
        self.model.train()
        total_loss = 0
        num_batches = len(dataloader)

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                logger.debug(f"Batch {batch_idx + 1}/{num_batches}, Loss: {loss.item():.4f}")

        return total_loss / num_batches

    def validate(self, dataloader: DataLoader) -> float:
        """
        Validate the model.

        Args:
            dataloader: DataLoader for validation data

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

        return total_loss / len(dataloader)

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        epochs: int = 50,
        save_path: Optional[str] = None,
        early_stopping_patience: int = 10,
    ) -> Dict[str, List[float]]:
        """
        Train the model.

        Args:
            train_dataloader: DataLoader for training
            val_dataloader: DataLoader for validation (optional)
            epochs: Number of epochs
            save_path: Path to save the best model
            early_stopping_patience: Patience for early stopping

        Returns:
            Training history
        """
        best_val_loss = float("inf")
        patience_counter = 0
        start_time = time.time()

        logger.info(f"Starting training on {self.device}")
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Batch size: {train_dataloader.batch_size}")
        logger.info(f"Training samples: {len(train_dataloader.dataset)}")
        if val_dataloader:
            logger.info(f"Validation samples: {len(val_dataloader.dataset)}")
        logger.info("-" * 50)

        for epoch in range(epochs):
            epoch_start = time.time()

            train_loss = self.train_epoch(train_dataloader)
            self.train_losses.append(train_loss)

            val_loss = None
            if val_dataloader:
                val_loss = self.validate(val_dataloader)
                self.val_losses.append(val_loss)
                self.scheduler.step(val_loss)

            if val_loss and val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if save_path:
                    self.save_model(save_path)
                    logger.info(f"Best model saved (Val Loss: {val_loss:.4f})")
            else:
                patience_counter += 1

            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.learning_rates.append(current_lr)

            logger.info(f"Epoch {epoch + 1}/{epochs} ({epoch_time:.1f}s)")
            logger.info(f"  Train Loss: {train_loss:.4f}")
            if val_loss:
                logger.info(f"  Val Loss: {val_loss:.4f}")
            logger.info(f"  LR: {current_lr:.6f}")
            logger.info(f"  Patience: {patience_counter}/{early_stopping_patience}")
            logger.info("-" * 30)

            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping after {epoch + 1} epochs")
                break

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.1f}s")

        if not val_dataloader and save_path:
            self.save_model(save_path)
            logger.info("Model saved at end of training (no validation)")

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "learning_rates": self.learning_rates,
        }

    def save_model(self, filepath: str) -> None:
        """
        Save the model.

        Args:
            filepath: Path to save the model
        """
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
                "learning_rates": self.learning_rates,
            },
            filepath,
        )

    def load_model(self, filepath: str) -> None:
        """
        Load a saved model.

        Args:
            filepath: Path to the file to load
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        self.learning_rates = checkpoint.get("learning_rates", [])

    def get_training_info(self) -> Dict:
        """
        Return the training information.

        Returns:
            Dictionary with training statistics
        """
        return {
            "device": self.device,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "weight_decay": self.optimizer.param_groups[0]["weight_decay"],
            "total_epochs": len(self.train_losses),
            "best_train_loss": min(self.train_losses) if self.train_losses else None,
            "best_val_loss": min(self.val_losses) if self.val_losses else None,
        }
