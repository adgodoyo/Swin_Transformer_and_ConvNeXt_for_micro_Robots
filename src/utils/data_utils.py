"""
Data utilities for pose classification

This module handles data splitting, dataloader creation, and class mappings.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict


def create_stratified_splits(
    dataset: torch.utils.data.Dataset,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create stratified train/val/test splits for dual-label dataset

    Maintains class distribution across splits by stratifying on combined
    pitch-roll configuration.

    Args:
        dataset: PyTorch dataset with dual labels
        train_ratio: Proportion for training (default: 0.6)
        val_ratio: Proportion for validation (default: 0.2)
        test_ratio: Proportion for testing (default: 0.2)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_indices, val_indices, test_indices)

    Example:
        >>> dataset = ClassificationDataset(...)
        >>> train_idx, val_idx, test_idx = create_stratified_splits(dataset)
        >>> train_subset = Subset(dataset, train_idx)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    indices = np.arange(len(dataset))

    # Get labels for stratification
    stratify_labels = []
    for idx in indices:
        _, labels = dataset[idx]
        # Combine pitch and roll into single label for stratification
        combined_label = labels['pitch'].item() * 100 + labels['roll'].item()
        stratify_labels.append(combined_label)

    # First split: train vs (val + test)
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(val_ratio + test_ratio),
        random_state=random_state,
        stratify=stratify_labels
    )

    # Second split: val vs test
    temp_labels = [stratify_labels[i] for i in temp_idx]
    val_size = val_ratio / (val_ratio + test_ratio)

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1 - val_size),
        random_state=random_state,
        stratify=temp_labels
    )

    return train_idx, val_idx, test_idx


def create_dataloaders(
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    test_dataset: torch.utils.data.Dataset,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for train/val/test splits

    Args:
        train_dataset: Training dataset (with augmentation)
        val_dataset: Validation dataset (without augmentation)
        test_dataset: Test dataset (without augmentation)
        train_indices: Indices for training split
        val_indices: Indices for validation split
        test_indices: Indices for test split
        batch_size: Batch size for all loaders
        num_workers: Number of worker processes (0 recommended for MPS)
        pin_memory: Pin memory for faster GPU transfer (False for MPS)

    Returns:
        Tuple of (train_loader, val_loader, test_loader)

    Example:
        >>> train_loader, val_loader, test_loader = create_dataloaders(
        ...     train_ds, val_ds, test_ds,
        ...     train_idx, val_idx, test_idx,
        ...     batch_size=64
        ... )
    """
    # Create subsets
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    test_subset = Subset(test_dataset, test_indices)

    # Create loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    return train_loader, val_loader, test_loader


def get_class_mappings(dataset: torch.utils.data.Dataset) -> Dict[str, Dict]:
    """
    Extract class mappings from dataset

    Args:
        dataset: Dataset with p_dict and r_dict attributes

    Returns:
        Dictionary containing:
            - 'pitch': {label_str: class_idx}
            - 'roll': {label_str: class_idx}
            - 'pitch_inv': {class_idx: label_str}
            - 'roll_inv': {class_idx: label_str}
            - 'num_pitch_classes': int
            - 'num_roll_classes': int

    Example:
        >>> mappings = get_class_mappings(dataset)
        >>> print(f"Pitch classes: {mappings['num_pitch_classes']}")
        >>> print(f"Class 0 is pitch {mappings['pitch_inv'][0]}")
    """
    pitch_dict = dataset.p_dict if hasattr(dataset, 'p_dict') else {}
    roll_dict = dataset.r_dict if hasattr(dataset, 'r_dict') else {}

    # Create inverse mappings
    pitch_inv = {v: k for k, v in pitch_dict.items()}
    roll_inv = {v: k for k, v in roll_dict.items()}

    return {
        'pitch': pitch_dict,
        'roll': roll_dict,
        'pitch_inv': pitch_inv,
        'roll_inv': roll_inv,
        'num_pitch_classes': len(pitch_dict),
        'num_roll_classes': len(roll_dict)
    }
