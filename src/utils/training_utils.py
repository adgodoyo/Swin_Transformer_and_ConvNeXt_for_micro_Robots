"""
Training utilities for pose classification models

This module provides core training functionality including:
- Metrics tracking across epochs
- Training and validation loops
- Prediction collection for analysis
- Early stopping mechanism
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Any, Optional


class MetricsTracker:
    """
    Track training metrics across epochs

    Stores history of losses, accuracies, and learning rates for both
    training and validation sets. Supports both single-head and dual-head models.

    Usage:
        tracker = MetricsTracker()
        tracker.update({
            'train_loss': 0.5,
            'train_pitch_acc': 0.85,
            'val_loss': 0.6
        })
        best_epoch, best_val = tracker.get_best_epoch('val_loss', mode='min')
    """

    def __init__(self):
        self.history = {
            # Total losses
            'train_loss': [],
            'val_loss': [],
            # Pitch-specific metrics
            'train_pitch_loss': [],
            'train_pitch_acc': [],
            'val_pitch_loss': [],
            'val_pitch_acc': [],
            # Roll-specific metrics
            'train_roll_loss': [],
            'train_roll_acc': [],
            'val_roll_loss': [],
            'val_roll_acc': [],
            # Learning rate
            'lr': []
        }

    def update(self, metrics: Dict[str, float]):
        """
        Update history with new metrics

        Args:
            metrics: Dictionary of metric names and values
        """
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)

    def get_best_epoch(self, metric: str = 'val_loss', mode: str = 'min') -> Tuple[int, float]:
        """
        Get the epoch with best performance for a given metric

        Args:
            metric: Name of metric to evaluate
            mode: 'min' for losses, 'max' for accuracies

        Returns:
            Tuple of (best_epoch_number, best_metric_value)
        """
        values = self.history[metric]
        if mode == 'min':
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)
        return best_idx + 1, values[best_idx]

    def get_history(self) -> Dict[str, list]:
        """Get complete history dictionary"""
        return self.history


class EarlyStopping:
    """
    Early stopping to prevent overfitting

    Monitors a metric and stops training when it stops improving.

    Usage:
        early_stop = EarlyStopping(patience=10, mode='max')
        for epoch in range(num_epochs):
            val_acc = validate(...)
            if early_stop(val_acc, model):
                print("Early stopping triggered")
                break
    """

    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 0.0,
        mode: str = 'min',
        save_path: Optional[str] = None
    ):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
            save_path: Path to save best model (optional)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.save_path = save_path

        self.counter = 0
        self.best_score = None
        self.early_stop = False

        if mode == 'min':
            self.compare = lambda score, best: score < (best - min_delta)
            self.best_score = float('inf')
        else:
            self.compare = lambda score, best: score > (best + min_delta)
            self.best_score = float('-inf')

    def __call__(self, score: float, model: Optional[torch.nn.Module] = None) -> bool:
        """
        Check if training should stop

        Args:
            score: Current metric value
            model: Model to save if improvement (optional)

        Returns:
            True if early stopping should trigger, False otherwise
        """
        if self.compare(score, self.best_score):
            self.best_score = score
            self.counter = 0

            # Save best model
            if model is not None and self.save_path is not None:
                torch.save(model.state_dict(), self.save_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


def train_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion_pitch: torch.nn.Module,
    criterion_roll: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None
) -> Dict[str, float]:
    """
    Train model for one epoch

    Performs forward pass, computes losses for both heads, backpropagation,
    and tracks metrics.

    Args:
        model: Neural network model with dual heads
        dataloader: Training data loader
        criterion_pitch: Loss function for pitch classification
        criterion_roll: Loss function for roll classification
        optimizer: Optimizer for parameter updates
        device: Device to run on (cuda/mps/cpu)
        scaler: GradScaler for mixed precision (optional)

    Returns:
        Dictionary containing:
            - loss: Combined loss
            - pitch_loss: Pitch-specific loss
            - roll_loss: Roll-specific loss
            - pitch_acc: Pitch classification accuracy
            - roll_acc: Roll classification accuracy
    """
    model.train()

    total_loss = 0
    pitch_loss_sum = 0
    roll_loss_sum = 0
    pitch_correct = 0
    roll_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        pitch_labels = labels['pitch'].to(device)
        roll_labels = labels['roll'].to(device)

        # Mixed precision forward pass
        if scaler is not None:
            with torch.cuda.amp.autocast():
                pitch_out, roll_out = model(images)
                loss_pitch = criterion_pitch(pitch_out, pitch_labels)
                loss_roll = criterion_roll(roll_out, roll_labels)
                loss = loss_pitch + loss_roll

            # Backward pass with scaling
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard forward pass
            pitch_out, roll_out = model(images)
            loss_pitch = criterion_pitch(pitch_out, pitch_labels)
            loss_roll = criterion_roll(roll_out, roll_labels)
            loss = loss_pitch + loss_roll

            # Standard backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Track metrics
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        pitch_loss_sum += loss_pitch.item() * batch_size
        roll_loss_sum += loss_roll.item() * batch_size

        pitch_pred = pitch_out.argmax(dim=1)
        roll_pred = roll_out.argmax(dim=1)
        pitch_correct += (pitch_pred == pitch_labels).sum().item()
        roll_correct += (roll_pred == roll_labels).sum().item()
        total_samples += batch_size

    metrics = {
        'loss': total_loss / total_samples,
        'pitch_loss': pitch_loss_sum / total_samples,
        'roll_loss': roll_loss_sum / total_samples,
        'pitch_acc': pitch_correct / total_samples,
        'roll_acc': roll_correct / total_samples
    }
    return metrics


@torch.no_grad()
def validate_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion_pitch: torch.nn.Module,
    criterion_roll: torch.nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """
    Validate model for one epoch

    Evaluates model without gradient computation.

    Args:
        model: Neural network model with dual heads
        dataloader: Validation data loader
        criterion_pitch: Loss function for pitch classification
        criterion_roll: Loss function for roll classification
        device: Device to run on (cuda/mps/cpu)

    Returns:
        Dictionary with same structure as train_epoch
    """
    model.eval()

    total_loss = 0
    pitch_loss_sum = 0
    roll_loss_sum = 0
    pitch_correct = 0
    roll_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        pitch_labels = labels['pitch'].to(device)
        roll_labels = labels['roll'].to(device)

        # Forward pass
        pitch_out, roll_out = model(images)

        # Compute losses
        loss_pitch = criterion_pitch(pitch_out, pitch_labels)
        loss_roll = criterion_roll(roll_out, roll_labels)
        loss = loss_pitch + loss_roll

        # Track metrics
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        pitch_loss_sum += loss_pitch.item() * batch_size
        roll_loss_sum += loss_roll.item() * batch_size

        pitch_pred = pitch_out.argmax(dim=1)
        roll_pred = roll_out.argmax(dim=1)
        pitch_correct += (pitch_pred == pitch_labels).sum().item()
        roll_correct += (roll_pred == roll_labels).sum().item()
        total_samples += batch_size

    metrics = {
        'loss': total_loss / total_samples,
        'pitch_loss': pitch_loss_sum / total_samples,
        'roll_loss': roll_loss_sum / total_samples,
        'pitch_acc': pitch_correct / total_samples,
        'roll_acc': roll_correct / total_samples
    }
    return metrics


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    return_features: bool = False
) -> Dict[str, np.ndarray]:
    """
    Collect all predictions and labels from a dataset

    Useful for detailed analysis, confusion matrices, and error analysis.

    Args:
        model: Neural network model
        dataloader: Data loader
        device: Device to run on
        return_features: If True, also return feature embeddings

    Returns:
        Dictionary containing:
            - pitch_preds: Predicted pitch classes
            - pitch_labels: True pitch classes
            - roll_preds: Predicted roll classes
            - roll_labels: True roll classes
            - pitch_probs: Pitch prediction probabilities
            - roll_probs: Roll prediction probabilities
            - features: Feature embeddings (if return_features=True)
    """
    model.eval()

    all_pitch_preds = []
    all_pitch_labels = []
    all_roll_preds = []
    all_roll_labels = []
    all_pitch_probs = []
    all_roll_probs = []
    all_features = [] if return_features else None

    for images, labels in dataloader:
        images = images.to(device)
        pitch_labels = labels['pitch']
        roll_labels = labels['roll']

        # Forward pass
        if return_features and hasattr(model, 'forward') and 'return_features' in model.forward.__code__.co_varnames:
            pitch_out, roll_out, features = model(images, return_features=True)
            all_features.extend(features.cpu().numpy())
        else:
            pitch_out, roll_out = model(images)

        # Compute probabilities
        pitch_probs = F.softmax(pitch_out, dim=1)
        roll_probs = F.softmax(roll_out, dim=1)

        # Collect predictions
        all_pitch_preds.extend(pitch_out.argmax(dim=1).cpu().numpy())
        all_pitch_labels.extend(pitch_labels.numpy())
        all_roll_preds.extend(roll_out.argmax(dim=1).cpu().numpy())
        all_roll_labels.extend(roll_labels.numpy())
        all_pitch_probs.extend(pitch_probs.cpu().numpy())
        all_roll_probs.extend(roll_probs.cpu().numpy())

    result = {
        'pitch_preds': np.array(all_pitch_preds),
        'pitch_labels': np.array(all_pitch_labels),
        'roll_preds': np.array(all_roll_preds),
        'roll_labels': np.array(all_roll_labels),
        'pitch_probs': np.array(all_pitch_probs),
        'roll_probs': np.array(all_roll_probs)
    }

    if return_features:
        result['features'] = np.array(all_features)

    return result
