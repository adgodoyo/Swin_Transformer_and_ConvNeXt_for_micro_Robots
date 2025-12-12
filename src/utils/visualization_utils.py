"""
Visualization utilities for model analysis

This module provides comprehensive visualization functions for:
- Training dynamics
- Confusion matrices
- Performance analysis
- Feature visualizations
- Attention maps
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from pathlib import Path
from typing import Dict, Optional, List
import torch
import torch.nn.functional as F
import optuna


def plot_training_curves(
    tracker_history: Dict[str, list],
    save_path: Optional[Path] = None,
    title: str = "Training Dynamics"
):
    """
    Plot comprehensive training curves

    Creates a multi-panel figure showing:
    - Total loss (train vs val)
    - Pitch loss and accuracy
    - Roll loss and accuracy
    - Learning rate schedule

    Args:
        tracker_history: Dictionary from MetricsTracker.get_history()
        save_path: Path to save figure (optional)
        title: Main title for the figure

    Example:
        >>> tracker = MetricsTracker()
        >>> # ... training loop ...
        >>> plot_training_curves(tracker.get_history(), save_path='curves.png')
    """
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    epochs = range(1, len(tracker_history['train_loss']) + 1)

    # Total loss
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(epochs, tracker_history['train_loss'], label='Train Loss', linewidth=2, color='blue')
    ax1.plot(epochs, tracker_history['val_loss'], label='Val Loss', linewidth=2, color='orange')
    ax1.set_xlabel('Epoch', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Loss', fontweight='bold', fontsize=12)
    ax1.set_title('Total Loss Evolution', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)

    # Pitch loss
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(epochs, tracker_history['train_pitch_loss'], label='Train', linewidth=2)
    ax2.plot(epochs, tracker_history['val_pitch_loss'], label='Val', linewidth=2)
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('Loss', fontweight='bold')
    ax2.set_title('Pitch Loss', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Pitch accuracy
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(epochs, tracker_history['train_pitch_acc'], label='Train', linewidth=2)
    ax3.plot(epochs, tracker_history['val_pitch_acc'], label='Val', linewidth=2)
    ax3.set_xlabel('Epoch', fontweight='bold')
    ax3.set_ylabel('Accuracy', fontweight='bold')
    ax3.set_title('Pitch Accuracy', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)

    # Roll loss
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(epochs, tracker_history['train_roll_loss'], label='Train', linewidth=2)
    ax4.plot(epochs, tracker_history['val_roll_loss'], label='Val', linewidth=2)
    ax4.set_xlabel('Epoch', fontweight='bold')
    ax4.set_ylabel('Loss', fontweight='bold')
    ax4.set_title('Roll Loss', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)

    # Roll accuracy
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(epochs, tracker_history['train_roll_acc'], label='Train', linewidth=2)
    ax5.plot(epochs, tracker_history['val_roll_acc'], label='Val', linewidth=2)
    ax5.set_xlabel('Epoch', fontweight='bold')
    ax5.set_ylabel('Accuracy', fontweight='bold')
    ax5.set_title('Roll Accuracy', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(alpha=0.3)

    # Learning rate
    ax6 = fig.add_subplot(gs[1:, 2])
    ax6.plot(epochs, tracker_history['lr'], color='red', linewidth=2)
    ax6.set_xlabel('Epoch', fontweight='bold')
    ax6.set_ylabel('Learning Rate', fontweight='bold')
    ax6.set_title('LR Schedule', fontsize=12, fontweight='bold')
    ax6.set_yscale('log')
    ax6.grid(alpha=0.3)

    plt.suptitle(title, fontsize=16, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_confusion_matrices(
    predictions: Dict[str, np.ndarray],
    class_names_pitch: List[str],
    class_names_roll: List[str],
    save_path: Optional[Path] = None,
    title: str = "Confusion Matrices"
):
    """
    Plot confusion matrices for pitch and roll predictions

    Args:
        predictions: Dictionary from collect_predictions()
        class_names_pitch: List of pitch class names (e.g., ['P0', 'P5', ...])
        class_names_roll: List of roll class names (e.g., ['R0', 'R5', ...])
        save_path: Path to save figure (optional)
        title: Main title for the figure

    Example:
        >>> preds = collect_predictions(model, test_loader, device)
        >>> plot_confusion_matrices(
        ...     preds,
        ...     ['P0', 'P5', 'P10'],
        ...     ['R0', 'R5'],
        ...     save_path='confusion.png'
        ... )
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Pitch confusion matrix
    pitch_cm = confusion_matrix(
        predictions['pitch_labels'],
        predictions['pitch_preds']
    )
    sns.heatmap(
        pitch_cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        ax=axes[0],
        xticklabels=class_names_pitch,
        yticklabels=class_names_pitch,
        cbar_kws={'label': 'Count'}
    )
    axes[0].set_xlabel('Predicted Pitch Class', fontweight='bold', fontsize=12)
    axes[0].set_ylabel('True Pitch Class', fontweight='bold', fontsize=12)
    axes[0].set_title('Pitch Confusion Matrix', fontsize=14, fontweight='bold')

    # Roll confusion matrix
    roll_cm = confusion_matrix(
        predictions['roll_labels'],
        predictions['roll_preds']
    )
    sns.heatmap(
        roll_cm,
        annot=True,
        fmt='d',
        cmap='Greens',
        ax=axes[1],
        xticklabels=class_names_roll,
        yticklabels=class_names_roll,
        cbar_kws={'label': 'Count'}
    )
    axes[1].set_xlabel('Predicted Roll Class', fontweight='bold', fontsize=12)
    axes[1].set_ylabel('True Roll Class', fontweight='bold', fontsize=12)
    axes[1].set_title('Roll Confusion Matrix', fontsize=14, fontweight='bold')

    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_per_class_performance(
    predictions: Dict[str, np.ndarray],
    class_names_pitch: List[str],
    class_names_roll: List[str],
    save_path: Optional[Path] = None
):
    """
    Plot per-class precision, recall, and F1 scores

    Args:
        predictions: Dictionary from collect_predictions()
        class_names_pitch: List of pitch class names
        class_names_roll: List of roll class names
        save_path: Path to save figure (optional)
    """
    from sklearn.metrics import precision_recall_fscore_support

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # Pitch per-class metrics
    pitch_precision, pitch_recall, pitch_f1, _ = precision_recall_fscore_support(
        predictions['pitch_labels'],
        predictions['pitch_preds'],
        average=None,
        zero_division=0
    )

    x = np.arange(len(class_names_pitch))
    width = 0.25

    axes[0].bar(x - width, pitch_precision, width, label='Precision', alpha=0.8)
    axes[0].bar(x, pitch_recall, width, label='Recall', alpha=0.8)
    axes[0].bar(x + width, pitch_f1, width, label='F1 Score', alpha=0.8)
    axes[0].set_xlabel('Pitch Class', fontweight='bold', fontsize=12)
    axes[0].set_ylabel('Score', fontweight='bold', fontsize=12)
    axes[0].set_title('Pitch: Per-Class Performance', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(class_names_pitch, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # Roll per-class metrics
    roll_precision, roll_recall, roll_f1, _ = precision_recall_fscore_support(
        predictions['roll_labels'],
        predictions['roll_preds'],
        average=None,
        zero_division=0
    )

    x = np.arange(len(class_names_roll))

    axes[1].bar(x - width, roll_precision, width, label='Precision', alpha=0.8, color='green')
    axes[1].bar(x, roll_recall, width, label='Recall', alpha=0.8, color='orange')
    axes[1].bar(x + width, roll_f1, width, label='F1 Score', alpha=0.8, color='red')
    axes[1].set_xlabel('Roll Class', fontweight='bold', fontsize=12)
    axes[1].set_ylabel('Score', fontweight='bold', fontsize=12)
    axes[1].set_title('Roll: Per-Class Performance', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(class_names_roll, rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_optuna_results(study, save_path: Optional[Path] = None):
    """
    Plot Optuna optimization results

    Creates visualization of:
    - Optimization history
    - Parameter importances
    - Parallel coordinate plot

    Args:
        study: Optuna study object
        save_path: Path to save figure (optional)
    """
    

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # Optimization history
    plt.sca(axes[0])
    optuna.visualization.matplotlib.plot_optimization_history(study)
    axes[0].set_title('Optimization History', fontsize=12, fontweight='bold')

    # Parameter importances
    plt.sca(axes[1])
    try:
        optuna.visualization.matplotlib.plot_param_importances(study)
        axes[1].set_title('Hyperparameter Importances', fontsize=12, fontweight='bold')
    except Exception:
        axes[1].text(0.5, 0.5, 'Not enough trials', ha='center', va='center')
        axes[1].set_title('Hyperparameter Importances', fontsize=12, fontweight='bold')

    # Parallel coordinate plot
    plt.sca(axes[2])
    try:
        optuna.visualization.matplotlib.plot_parallel_coordinate(study)
        axes[2].set_title('Parallel Coordinate Plot', fontsize=12, fontweight='bold')
    except Exception:
        axes[2].text(0.5, 0.5, 'Not enough trials', ha='center', va='center')
        axes[2].set_title('Parallel Coordinate Plot', fontsize=12, fontweight='bold')


    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_feature_maps(
    model: torch.nn.Module,
    image: torch.Tensor,
    layer_name: str,
    device: torch.device,
    save_path: Optional[Path] = None,
    num_filters: int = 16
):
    """
    Visualize feature maps from a specific layer

    Args:
        model: PyTorch model
        image: Input image tensor (1, C, H, W)
        layer_name: Name of layer to visualize
        device: Device to run on
        save_path: Path to save figure (optional)
        num_filters: Number of filters to display
    """
    model.eval()

    # Hook to capture activations
    activations = {}

    def hook_fn(module, input, output):
        activations['output'] = output.detach()

    # Register hook
    target_layer = dict(model.named_modules())[layer_name]
    hook = target_layer.register_forward_hook(hook_fn)

    # Forward pass
    image = image.to(device)
    with torch.no_grad():
        _ = model(image)

    # Remove hook
    hook.remove()

    # Get activations
    acts = activations['output'][0].cpu().numpy()  # (C, H, W)

    # Plot
    num_filters = min(num_filters, acts.shape[0])
    grid_size = int(np.ceil(np.sqrt(num_filters)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten()

    for i in range(num_filters):
        axes[i].imshow(acts[i], cmap='viridis')
        axes[i].axis('off')
        axes[i].set_title(f'Filter {i}', fontsize=8)

    for i in range(num_filters, len(axes)):
        axes[i].axis('off')

    plt.suptitle(f'Feature Maps: {layer_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_grad_cam(
    model: torch.nn.Module,
    image: torch.Tensor,
    target_layer_name: str,
    class_idx: int,
    device: torch.device,
    save_path: Optional[Path] = None
):
    """
    Generate and plot Grad-CAM heatmap

    Args:
        model: PyTorch model
        image: Input image tensor (1, C, H, W)
        target_layer_name: Name of target layer for Grad-CAM
        class_idx: Target class index for visualization
        device: Device to run on
        save_path: Path to save figure (optional)
    """
    model.eval()

    # Storage for gradients and activations
    gradients = {}
    activations = {}

    def backward_hook(module, grad_input, grad_output):
        gradients['value'] = grad_output[0]

    def forward_hook(module, input, output):
        activations['value'] = output

    # Register hooks
    target_layer = dict(model.named_modules())[target_layer_name]
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    # Forward pass
    image = image.to(device)
    image.requires_grad = True
    output = model(image)

    # If dual-head, select pitch output
    if isinstance(output, tuple):
        output = output[0]

    # Backward pass
    model.zero_grad()
    output[0, class_idx].backward()

    # Remove hooks
    forward_handle.remove()
    backward_handle.remove()

    # Compute Grad-CAM
    grads = gradients['value'].cpu().data.numpy()[0]
    acts = activations['value'].cpu().data.numpy()[0]

    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)

    # Resize to image size
    from PIL import Image as PILImage
    cam_resized = PILImage.fromarray((cam * 255).astype(np.uint8))
    cam_resized = cam_resized.resize((image.shape[3], image.shape[2]), PILImage.BILINEAR)
    cam_resized = np.array(cam_resized) / 255.0

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    img_display = image[0].detach().cpu().permute(1, 2, 0).numpy()
    img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())
    axes[0].imshow(img_display)
    axes[0].set_title('Original Image', fontweight='bold')
    axes[0].axis('off')

    # Heatmap
    axes[1].imshow(cam_resized, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap', fontweight='bold')
    axes[1].axis('off')

    # Overlay
    axes[2].imshow(img_display)
    axes[2].imshow(cam_resized, cmap='jet', alpha=0.5)
    axes[2].set_title('Overlay', fontweight='bold')
    axes[2].axis('off')

    plt.suptitle(f'Grad-CAM: Class {class_idx}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_embedding_tsne(
    features: np.ndarray,
    labels: np.ndarray,
    label_names: List[str],
    save_path: Optional[Path] = None,
    title: str = "t-SNE Embedding"
):
    """
    Plot t-SNE visualization of learned features

    Args:
        features: Feature embeddings (N, D)
        labels: Class labels (N,)
        label_names: List of class names
        save_path: Path to save figure (optional)
        title: Figure title
    """
    # Compute t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features)

    # Plot
    plt.figure(figsize=(12, 10))

    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        mask = labels == label
        plt.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=[color],
            label=label_names[label] if label < len(label_names) else f'Class {label}',
            alpha=0.7,
            s=30,
            edgecolors='k',
            linewidth=0.3
        )

    plt.xlabel('t-SNE Component 1', fontweight='bold', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontweight='bold', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
