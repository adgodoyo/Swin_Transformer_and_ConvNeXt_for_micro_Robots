"""
Regression utilities for depth estimation tasks

This module provides comprehensive utilities for depth regression including:
- Training and validation functions
- Metrics computation and tracking
- Visualization functions for regression analysis
- Transfer learning helpers for loading pretrained backbones
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from typing import Dict, Tuple, Any, Optional, List
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch_regression(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None
) -> Dict[str, float]:
    """
    Train model for one epoch on regression task

    Performs forward pass, computes loss, backpropagation, and tracks metrics.

    Args:
        model: Neural network model for regression
        dataloader: Training data loader
        criterion: Loss function (e.g., MSELoss, L1Loss, HuberLoss)
        optimizer: Optimizer for parameter updates
        device: Device to run on (cuda/mps/cpu)
        scaler: GradScaler for mixed precision training (optional)

    Returns:
        Dictionary containing:
            - loss: Average loss over epoch
            - rmse: Root mean squared error
            - mae: Mean absolute error

    Example:
        >>> criterion = nn.MSELoss()
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        >>> metrics = train_epoch_regression(
        ...     model, train_loader, criterion, optimizer, device
        ... )
        >>> print(f"Loss: {metrics['loss']:.4f}, RMSE: {metrics['rmse']:.4f}")
    """
    model.train()

    total_loss = 0.0
    total_squared_error = 0.0
    total_absolute_error = 0.0
    total_samples = 0

    for images, depths in dataloader:
        images = images.to(device)
        depths = depths.to(device)

        # Ensure depths are the right shape (batch_size, 1)
        if depths.dim() == 1:
            depths = depths.unsqueeze(1)

        # Mixed precision forward pass
        if scaler is not None:
            with torch.cuda.amp.autocast():
                predictions = model(images)
                loss = criterion(predictions, depths)

            # Backward pass with scaling
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard forward pass
            predictions = model(images)
            loss = criterion(predictions, depths)

            # Standard backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Track metrics
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size

        # Compute additional metrics
        with torch.no_grad():
            squared_error = ((predictions - depths) ** 2).sum().item()
            absolute_error = torch.abs(predictions - depths).sum().item()

            total_squared_error += squared_error
            total_absolute_error += absolute_error
            total_samples += batch_size

    # Compute epoch metrics
    avg_loss = total_loss / total_samples
    rmse = np.sqrt(total_squared_error / total_samples)
    mae = total_absolute_error / total_samples

    metrics = {
        'loss': avg_loss,
        'rmse': rmse,
        'mae': mae
    }

    return metrics


@torch.no_grad()
def validate_epoch_regression(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """
    Validate model for one epoch on regression task

    Evaluates model without gradient computation.

    Args:
        model: Neural network model for regression
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to run on (cuda/mps/cpu)

    Returns:
        Dictionary with same structure as train_epoch_regression

    Example:
        >>> val_metrics = validate_epoch_regression(
        ...     model, val_loader, criterion, device
        ... )
        >>> print(f"Val Loss: {val_metrics['loss']:.4f}")
    """
    model.eval()

    total_loss = 0.0
    total_squared_error = 0.0
    total_absolute_error = 0.0
    total_samples = 0

    for images, depths in dataloader:
        images = images.to(device)
        depths = depths.to(device)

        # Ensure depths are the right shape
        if depths.dim() == 1:
            depths = depths.unsqueeze(1)

        # Forward pass
        predictions = model(images)
        loss = criterion(predictions, depths)

        # Track metrics
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size

        squared_error = ((predictions - depths) ** 2).sum().item()
        absolute_error = torch.abs(predictions - depths).sum().item()

        total_squared_error += squared_error
        total_absolute_error += absolute_error
        total_samples += batch_size

    # Compute epoch metrics
    avg_loss = total_loss / total_samples
    rmse = np.sqrt(total_squared_error / total_samples)
    mae = total_absolute_error / total_samples

    metrics = {
        'loss': avg_loss,
        'rmse': rmse,
        'mae': mae
    }

    return metrics


@torch.no_grad()
def collect_predictions_regression(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    return_images: bool = False
) -> Dict[str, np.ndarray]:
    """
    Collect all predictions and ground truth from a dataset

    Useful for detailed analysis, visualization, and error analysis.

    Args:
        model: Neural network model
        dataloader: Data loader
        device: Device to run on
        return_images: If True, also return the input images

    Returns:
        Dictionary containing:
            - predictions: Predicted depth values (N,)
            - targets: True depth values (N,)
            - errors: Prediction errors (predictions - targets) (N,)
            - absolute_errors: Absolute errors (N,)
            - squared_errors: Squared errors (N,)
            - images: Input images (N, C, H, W) if return_images=True

    Example:
        >>> preds = collect_predictions_regression(model, test_loader, device)
        >>> print(f"Mean error: {preds['errors'].mean():.4f}")
        >>> print(f"Median absolute error: {np.median(preds['absolute_errors']):.4f}")
    """
    model.eval()

    all_predictions = []
    all_targets = []
    all_images = [] if return_images else None

    for images, depths in dataloader:
        images = images.to(device)
        depths = depths.to(device)

        # Ensure depths are the right shape
        if depths.dim() == 1:
            depths = depths.unsqueeze(1)

        # Forward pass
        predictions = model(images)

        # Store predictions and targets
        all_predictions.append(predictions.cpu().numpy())
        all_targets.append(depths.cpu().numpy())

        if return_images:
            all_images.append(images.cpu().numpy())

    # Concatenate all batches
    predictions = np.concatenate(all_predictions, axis=0).flatten()
    targets = np.concatenate(all_targets, axis=0).flatten()

    # Compute errors
    errors = predictions - targets
    absolute_errors = np.abs(errors)
    squared_errors = errors ** 2

    result = {
        'predictions': predictions,
        'targets': targets,
        'errors': errors,
        'absolute_errors': absolute_errors,
        'squared_errors': squared_errors
    }

    if return_images:
        result['images'] = np.concatenate(all_images, axis=0)

    return result


# ============================================================================
# METRICS FUNCTIONS
# ============================================================================

def compute_regression_metrics(
    predictions: np.ndarray,
    targets: np.ndarray
) -> Dict[str, float]:
    """
    Compute comprehensive regression metrics

    Args:
        predictions: Predicted values (N,)
        targets: True values (N,)

    Returns:
        Dictionary containing:
            - mse: Mean squared error
            - rmse: Root mean squared error
            - mae: Mean absolute error
            - median_ae: Median absolute error
            - r2: R-squared coefficient of determination
            - mape: Mean absolute percentage error (if no zeros in targets)
            - max_error: Maximum absolute error
            - std_error: Standard deviation of errors

    Example:
        >>> preds = collect_predictions_regression(model, test_loader, device)
        >>> metrics = compute_regression_metrics(preds['predictions'], preds['targets'])
        >>> print(f"R²: {metrics['r2']:.4f}")
        >>> print(f"RMSE: {metrics['rmse']:.4f}")
    """
    # Ensure arrays are 1D
    predictions = predictions.flatten()
    targets = targets.flatten()

    # Compute errors
    errors = predictions - targets
    absolute_errors = np.abs(errors)
    squared_errors = errors ** 2

    # Basic metrics
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets, predictions)
    median_ae = np.median(absolute_errors)
    r2 = r2_score(targets, predictions)
    max_error = np.max(absolute_errors)
    std_error = np.std(errors)

    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'median_ae': median_ae,
        'r2': r2,
        'max_error': max_error,
        'std_error': std_error
    }

    # MAPE (only if no zeros in targets)
    if not np.any(targets == 0):
        mape = np.mean(np.abs((targets - predictions) / targets)) * 100
        metrics['mape'] = mape

    return metrics


class RegressionMetricsTracker:
    """
    Track regression metrics across epochs

    Stores history of losses and metrics for both training and validation sets.

    Usage:
        tracker = RegressionMetricsTracker()
        tracker.update({
            'train_loss': 0.5,
            'train_rmse': 0.7,
            'val_loss': 0.6,
            'val_rmse': 0.75
        })
        best_epoch, best_val = tracker.get_best_epoch('val_loss', mode='min')
    """

    def __init__(self):
        self.history = {
            # Training metrics
            'train_loss': [],
            'train_rmse': [],
            'train_mae': [],

            # Validation metrics
            'val_loss': [],
            'val_rmse': [],
            'val_mae': [],

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

    def get_best_epoch(
        self,
        metric: str = 'val_loss',
        mode: str = 'min'
    ) -> Tuple[int, float]:
        """
        Get the epoch with best performance for a given metric

        Args:
            metric: Name of metric to evaluate
            mode: 'min' for loss/error metrics, 'max' for R²

        Returns:
            Tuple of (best_epoch_number, best_metric_value)
        """
        values = self.history[metric]
        if not values:
            return 0, float('inf') if mode == 'min' else float('-inf')

        if mode == 'min':
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)

        return best_idx + 1, values[best_idx]

    def get_history(self) -> Dict[str, list]:
        """Get complete history dictionary"""
        return self.history


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_regression_training_curves(
    tracker_history: Dict[str, list],
    save_path: Optional[Path] = None,
    title: str = "Regression Training Dynamics"
):
    """
    Plot comprehensive training curves for regression

    Creates a multi-panel figure showing:
    - Loss evolution (train vs val)
    - RMSE evolution (train vs val)
    - MAE evolution (train vs val)
    - Learning rate schedule

    Args:
        tracker_history: Dictionary from RegressionMetricsTracker.get_history()
        save_path: Path to save figure (optional)
        title: Main title for the figure

    Example:
        >>> tracker = RegressionMetricsTracker()
        >>> # ... training loop ...
        >>> plot_regression_training_curves(
        ...     tracker.get_history(),
        ...     save_path='training_curves.png'
        ... )
    """
    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    epochs = range(1, len(tracker_history['train_loss']) + 1)

    # Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, tracker_history['train_loss'],
             label='Train Loss', linewidth=2, color='blue', marker='o')
    ax1.plot(epochs, tracker_history['val_loss'],
             label='Val Loss', linewidth=2, color='orange', marker='s')
    ax1.set_xlabel('Epoch', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Loss', fontweight='bold', fontsize=12)
    ax1.set_title('Loss Evolution', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)

    # RMSE
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, tracker_history['train_rmse'],
             label='Train RMSE', linewidth=2, color='green', marker='o')
    ax2.plot(epochs, tracker_history['val_rmse'],
             label='Val RMSE', linewidth=2, color='red', marker='s')
    ax2.set_xlabel('Epoch', fontweight='bold', fontsize=12)
    ax2.set_ylabel('RMSE', fontweight='bold', fontsize=12)
    ax2.set_title('RMSE Evolution', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)

    # MAE
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(epochs, tracker_history['train_mae'],
             label='Train MAE', linewidth=2, color='purple', marker='o')
    ax3.plot(epochs, tracker_history['val_mae'],
             label='Val MAE', linewidth=2, color='brown', marker='s')
    ax3.set_xlabel('Epoch', fontweight='bold', fontsize=12)
    ax3.set_ylabel('MAE', fontweight='bold', fontsize=12)
    ax3.set_title('MAE Evolution', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(alpha=0.3)

    # Learning rate
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(epochs, tracker_history['lr'],
             color='red', linewidth=2, marker='d')
    ax4.set_xlabel('Epoch', fontweight='bold', fontsize=12)
    ax4.set_ylabel('Learning Rate', fontweight='bold', fontsize=12)
    ax4.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax4.set_yscale('log')
    ax4.grid(alpha=0.3)

    plt.suptitle(title, fontsize=16, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_predictions_vs_actual(
    predictions: np.ndarray,
    targets: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "Predictions vs Actual Values"
):
    """
    Scatter plot of predictions vs ground truth

    Args:
        predictions: Predicted depth values (N,)
        targets: True depth values (N,)
        save_path: Path to save figure (optional)
        title: Figure title

    Example:
        >>> preds = collect_predictions_regression(model, test_loader, device)
        >>> plot_predictions_vs_actual(
        ...     preds['predictions'],
        ...     preds['targets'],
        ...     save_path='pred_vs_actual.png'
        ... )
    """
    predictions = predictions.flatten()
    targets = targets.flatten()

    # Compute R²
    r2 = r2_score(targets, predictions)

    # Create plot
    plt.figure(figsize=(10, 10))

    # Scatter plot
    plt.scatter(targets, predictions, alpha=0.5, s=30, edgecolors='k', linewidth=0.3)

    # Perfect prediction line
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val],
             'r--', linewidth=2, label='Perfect Prediction')

    # Add regression line
    z = np.polyfit(targets, predictions, 1)
    p = np.poly1d(z)
    plt.plot(targets, p(targets), 'b-', alpha=0.8, linewidth=2,
             label=f'Fit: y={z[0]:.3f}x+{z[1]:.3f}')

    plt.xlabel('Actual Depth', fontweight='bold', fontsize=12)
    plt.ylabel('Predicted Depth', fontweight='bold', fontsize=12)
    plt.title(f'{title}\nR² = {r2:.4f}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.axis('equal')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_residuals(
    predictions: np.ndarray,
    targets: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "Residual Plot"
):
    """
    Plot residuals (errors) vs predicted values

    Helps identify patterns in prediction errors and assess model bias.

    Args:
        predictions: Predicted depth values (N,)
        targets: True depth values (N,)
        save_path: Path to save figure (optional)
        title: Figure title

    Example:
        >>> preds = collect_predictions_regression(model, test_loader, device)
        >>> plot_residuals(
        ...     preds['predictions'],
        ...     preds['targets'],
        ...     save_path='residuals.png'
        ... )
    """
    predictions = predictions.flatten()
    targets = targets.flatten()
    residuals = predictions - targets

    plt.figure(figsize=(12, 6))

    # Residual plot
    plt.scatter(predictions, residuals, alpha=0.5, s=30, edgecolors='k', linewidth=0.3)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero Error')

    # Add horizontal lines at ±1 std
    std_resid = np.std(residuals)
    plt.axhline(y=std_resid, color='orange', linestyle=':', linewidth=1.5,
                label=f'±1 Std ({std_resid:.3f})')
    plt.axhline(y=-std_resid, color='orange', linestyle=':', linewidth=1.5)

    plt.xlabel('Predicted Depth', fontweight='bold', fontsize=12)
    plt.ylabel('Residuals (Predicted - Actual)', fontweight='bold', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_error_distribution(
    predictions: np.ndarray,
    targets: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "Error Distribution"
):
    """
    Histogram of prediction errors

    Visualizes the distribution of errors to assess model performance.

    Args:
        predictions: Predicted depth values (N,)
        targets: True depth values (N,)
        save_path: Path to save figure (optional)
        title: Figure title

    Example:
        >>> preds = collect_predictions_regression(model, test_loader, device)
        >>> plot_error_distribution(
        ...     preds['predictions'],
        ...     preds['targets'],
        ...     save_path='error_dist.png'
        ... )
    """
    predictions = predictions.flatten()
    targets = targets.flatten()
    errors = predictions - targets

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Histogram of errors
    axes[0].hist(errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0].axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    axes[0].axvline(x=np.mean(errors), color='green', linestyle='-', linewidth=2,
                    label=f'Mean Error: {np.mean(errors):.4f}')
    axes[0].set_xlabel('Error (Predicted - Actual)', fontweight='bold', fontsize=12)
    axes[0].set_ylabel('Frequency', fontweight='bold', fontsize=12)
    axes[0].set_title('Error Distribution', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.3)

    # Histogram of absolute errors
    abs_errors = np.abs(errors)
    axes[1].hist(abs_errors, bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[1].axvline(x=np.mean(abs_errors), color='red', linestyle='-', linewidth=2,
                    label=f'Mean AE: {np.mean(abs_errors):.4f}')
    axes[1].axvline(x=np.median(abs_errors), color='green', linestyle='--', linewidth=2,
                    label=f'Median AE: {np.median(abs_errors):.4f}')
    axes[1].set_xlabel('Absolute Error', fontweight='bold', fontsize=12)
    axes[1].set_ylabel('Frequency', fontweight='bold', fontsize=12)
    axes[1].set_title('Absolute Error Distribution', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(alpha=0.3)

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_depth_predictions_gallery(
    images: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    num_samples: int = 16,
    save_path: Optional[Path] = None,
    title: str = "Depth Predictions Gallery"
):
    """
    Gallery of images with predicted and actual depths

    Args:
        images: Input images (N, C, H, W)
        predictions: Predicted depth values (N,)
        targets: True depth values (N,)
        num_samples: Number of samples to display
        save_path: Path to save figure (optional)
        title: Figure title

    Example:
        >>> preds = collect_predictions_regression(
        ...     model, test_loader, device, return_images=True
        ... )
        >>> plot_depth_predictions_gallery(
        ...     preds['images'],
        ...     preds['predictions'],
        ...     preds['targets'],
        ...     num_samples=16,
        ...     save_path='gallery.png'
        ... )
    """
    predictions = predictions.flatten()
    targets = targets.flatten()

    # Select random samples
    num_samples = min(num_samples, len(images))
    indices = np.random.choice(len(images), num_samples, replace=False)

    # Compute grid size
    grid_size = int(np.ceil(np.sqrt(num_samples)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        img = images[idx]
        pred = predictions[idx]
        target = targets[idx]
        error = pred - target

        # Denormalize image for display
        img_display = img.transpose(1, 2, 0)  # CHW -> HWC
        img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())

        axes[i].imshow(img_display)
        axes[i].axis('off')

        # Color based on error magnitude
        error_pct = abs(error / target) * 100 if target != 0 else float('inf')
        color = 'green' if error_pct < 10 else 'orange' if error_pct < 20 else 'red'

        axes[i].set_title(
            f'Pred: {pred:.3f}\nActual: {target:.3f}\nError: {error:.3f}',
            fontsize=10,
            fontweight='bold',
            color=color
        )

    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# ============================================================================
# TRANSFER LEARNING HELPERS
# ============================================================================

def load_classification_backbone(
    classification_model_path: str,
    regression_model: nn.Module,
    device: torch.device,
    strict: bool = False
) -> nn.Module:
    """
    Load pretrained classification model backbone into regression model

    Transfers feature extraction layers from a pretrained classification model
    to initialize a regression model. Useful for transfer learning.

    Args:
        classification_model_path: Path to saved classification model checkpoint
        regression_model: Regression model to initialize
        device: Device to load model on
        strict: If True, requires exact key matching in state dict

    Returns:
        Regression model with loaded backbone weights

    Example:
        >>> regression_model = DepthRegressionModel()
        >>> regression_model = load_classification_backbone(
        ...     'checkpoints/best_classifier.pth',
        ...     regression_model,
        ...     device,
        ...     strict=False
        ... )
    """
    # Load classification checkpoint
    checkpoint = torch.load(classification_model_path, map_location=device)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Filter out keys that don't match (e.g., final classification layers)
    model_dict = regression_model.state_dict()

    # Only load matching keys
    pretrained_dict = {}
    for k, v in state_dict.items():
        # Remove 'module.' prefix if present (from DataParallel)
        k_clean = k.replace('module.', '')

        if k_clean in model_dict and v.shape == model_dict[k_clean].shape:
            pretrained_dict[k_clean] = v

    # Update regression model
    model_dict.update(pretrained_dict)
    regression_model.load_state_dict(model_dict, strict=strict)

    print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} layers from classification model")

    return regression_model


def freeze_backbone(
    model: nn.Module,
    freeze_until: Optional[str] = None
) -> nn.Module:
    """
    Freeze backbone layers for transfer learning

    Freezes all parameters up to (and including) a specified layer.
    If no layer is specified, freezes all feature extraction layers.

    Args:
        model: Neural network model
        freeze_until: Name of last layer to freeze (optional)
                     If None, freezes all layers except final regression head

    Returns:
        Model with frozen backbone

    Example:
        >>> # Freeze all convolutional layers
        >>> model = freeze_backbone(model, freeze_until='conv5')
        >>>
        >>> # Freeze everything except final layer
        >>> model = freeze_backbone(model)
    """
    freeze_all = freeze_until is None
    found_target = False

    for name, param in model.named_parameters():
        if freeze_all:
            # Freeze all except regression head (fc, regressor, head, etc.)
            if any(x in name.lower() for x in ['fc', 'regressor', 'head', 'regression']):
                param.requires_grad = True
            else:
                param.requires_grad = False
        else:
            # Freeze until target layer
            if not found_target:
                param.requires_grad = False
                if freeze_until in name:
                    found_target = True
            else:
                param.requires_grad = True

    # Count frozen/trainable parameters
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Frozen parameters: {frozen_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    return model


def unfreeze_backbone(
    model: nn.Module,
    unfreeze_from: Optional[str] = None
) -> nn.Module:
    """
    Unfreeze backbone layers for fine-tuning

    Unfreezes parameters starting from a specified layer.
    If no layer is specified, unfreezes all parameters.

    Args:
        model: Neural network model
        unfreeze_from: Name of first layer to unfreeze (optional)
                      If None, unfreezes all layers

    Returns:
        Model with unfrozen layers

    Example:
        >>> # Unfreeze from conv4 onwards
        >>> model = unfreeze_backbone(model, unfreeze_from='conv4')
        >>>
        >>> # Unfreeze all layers
        >>> model = unfreeze_backbone(model)
    """
    if unfreeze_from is None:
        # Unfreeze all
        for param in model.parameters():
            param.requires_grad = True
    else:
        # Unfreeze from target layer onwards
        found_target = False
        for name, param in model.named_parameters():
            if not found_target:
                if unfreeze_from in name:
                    found_target = True
                    param.requires_grad = True
            else:
                param.requires_grad = True

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Trainable parameters: {trainable_params:,}/{total_params:,}")

    return model
