"""
Evaluation utilities for model analysis

This module provides functions for:
- Metric computation
- Classification reports
- Error analysis
- Computational cost estimation
"""

import numpy as np
import time
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)
from typing import Dict, List, Tuple


def compute_metrics(
    predictions: Dict[str, np.ndarray],
    average: str = 'weighted'
) -> Dict[str, Dict[str, float]]:
    """
    Compute comprehensive classification metrics

    Args:
        predictions: Dictionary from collect_predictions()
        average: Averaging method ('micro', 'macro', 'weighted')

    Returns:
        Dictionary containing metrics for pitch and roll:
            - accuracy
            - precision
            - recall
            - f1_score

    Example:
        >>> preds = collect_predictions(model, test_loader, device)
        >>> metrics = compute_metrics(preds)
        >>> print(f"Pitch accuracy: {metrics['pitch']['accuracy']:.4f}")
    """
    # Pitch metrics
    pitch_acc = accuracy_score(
        predictions['pitch_labels'],
        predictions['pitch_preds']
    )

    pitch_precision, pitch_recall, pitch_f1, _ = precision_recall_fscore_support(
        predictions['pitch_labels'],
        predictions['pitch_preds'],
        average=average,
        zero_division=0
    )

    # Roll metrics
    roll_acc = accuracy_score(
        predictions['roll_labels'],
        predictions['roll_preds']
    )

    roll_precision, roll_recall, roll_f1, _ = precision_recall_fscore_support(
        predictions['roll_labels'],
        predictions['roll_preds'],
        average=average,
        zero_division=0
    )

    return {
        'pitch': {
            'accuracy': pitch_acc,
            'precision': pitch_precision,
            'recall': pitch_recall,
            'f1_score': pitch_f1
        },
        'roll': {
            'accuracy': roll_acc,
            'precision': roll_precision,
            'recall': roll_recall,
            'f1_score': roll_f1
        }
    }


def generate_classification_report(
    predictions: Dict[str, np.ndarray],
    class_names_pitch: List[str],
    class_names_roll: List[str]
) -> Tuple[str, str]:
    """
    Generate detailed classification reports

    Args:
        predictions: Dictionary from collect_predictions()
        class_names_pitch: List of pitch class names
        class_names_roll: List of roll class names

    Returns:
        Tuple of (pitch_report, roll_report) as strings

    Example:
        >>> pitch_report, roll_report = generate_classification_report(
        ...     preds, ['P0', 'P5'], ['R0', 'R5']
        ... )
        >>> print(pitch_report)
    """
    pitch_report = classification_report(
        predictions['pitch_labels'],
        predictions['pitch_preds'],
        target_names=class_names_pitch,
        zero_division=0
    )

    roll_report = classification_report(
        predictions['roll_labels'],
        predictions['roll_preds'],
        target_names=class_names_roll,
        zero_division=0
    )

    return pitch_report, roll_report


def analyze_errors(
    predictions: Dict[str, np.ndarray],
    class_names_pitch: List[str],
    class_names_roll: List[str],
    top_k: int = 10
) -> Dict[str, any]:
    """
    Analyze prediction errors

    Identifies most confused classes and worst predictions.

    Args:
        predictions: Dictionary from collect_predictions()
        class_names_pitch: List of pitch class names
        class_names_roll: List of roll class names
        top_k: Number of top errors to return

    Returns:
        Dictionary containing:
            - pitch_error_indices: Indices of misclassified pitch samples
            - roll_error_indices: Indices of misclassified roll samples
            - pitch_confusion_pairs: Most confused pitch class pairs
            - roll_confusion_pairs: Most confused roll class pairs
            - low_confidence_pitch: Indices of low-confidence pitch predictions
            - low_confidence_roll: Indices of low-confidence roll predictions

    Example:
        >>> errors = analyze_errors(preds, pitch_names, roll_names)
        >>> print(f"Pitch errors: {len(errors['pitch_error_indices'])}")
    """
    # Find misclassified samples
    pitch_errors = predictions['pitch_preds'] != predictions['pitch_labels']
    roll_errors = predictions['roll_preds'] != predictions['roll_labels']

    pitch_error_idx = np.where(pitch_errors)[0]
    roll_error_idx = np.where(roll_errors)[0]

    # Find most confused class pairs
    from sklearn.metrics import confusion_matrix

    pitch_cm = confusion_matrix(
        predictions['pitch_labels'],
        predictions['pitch_preds']
    )

    roll_cm = confusion_matrix(
        predictions['roll_labels'],
        predictions['roll_preds']
    )

    # Get top confused pairs (excluding diagonal)
    pitch_confusion_pairs = []
    for i in range(pitch_cm.shape[0]):
        for j in range(pitch_cm.shape[1]):
            if i != j and pitch_cm[i, j] > 0:
                pitch_confusion_pairs.append({
                    'true_class': class_names_pitch[i],
                    'pred_class': class_names_pitch[j],
                    'count': pitch_cm[i, j]
                })

    pitch_confusion_pairs = sorted(
        pitch_confusion_pairs,
        key=lambda x: x['count'],
        reverse=True
    )[:top_k]

    roll_confusion_pairs = []
    for i in range(roll_cm.shape[0]):
        for j in range(roll_cm.shape[1]):
            if i != j and roll_cm[i, j] > 0:
                roll_confusion_pairs.append({
                    'true_class': class_names_roll[i],
                    'pred_class': class_names_roll[j],
                    'count': roll_cm[i, j]
                })

    roll_confusion_pairs = sorted(
        roll_confusion_pairs,
        key=lambda x: x['count'],
        reverse=True
    )[:top_k]

    # Find low-confidence predictions
    pitch_max_probs = predictions['pitch_probs'].max(axis=1)
    roll_max_probs = predictions['roll_probs'].max(axis=1)

    # Get indices of lowest confidence predictions
    low_conf_pitch_idx = np.argsort(pitch_max_probs)[:top_k]
    low_conf_roll_idx = np.argsort(roll_max_probs)[:top_k]

    return {
        'pitch_error_indices': pitch_error_idx,
        'roll_error_indices': roll_error_idx,
        'pitch_confusion_pairs': pitch_confusion_pairs,
        'roll_confusion_pairs': roll_confusion_pairs,
        'low_confidence_pitch': low_conf_pitch_idx,
        'low_confidence_roll': low_conf_roll_idx,
        'pitch_error_rate': len(pitch_error_idx) / len(predictions['pitch_labels']),
        'roll_error_rate': len(roll_error_idx) / len(predictions['roll_labels'])
    }


def compute_computational_cost(
    model: torch.nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
    device: torch.device = torch.device('cpu'),
    num_runs: int = 100
) -> Dict[str, any]:
    """
    Compute computational cost metrics for a model

    Measures:
    - Number of parameters
    - Inference time (mean and std)
    - Memory footprint

    Args:
        model: PyTorch model
        input_size: Input tensor size (batch, channels, height, width)
        device: Device to run on
        num_runs: Number of inference runs for timing

    Returns:
        Dictionary containing:
            - num_parameters: Total trainable parameters
            - num_parameters_m: Parameters in millions
            - inference_time_mean: Mean inference time (ms)
            - inference_time_std: Std of inference time (ms)
            - memory_mb: Approximate model memory (MB)

    Example:
        >>> cost = compute_computational_cost(model, device=device)
        >>> print(f"Parameters: {cost['num_parameters_m']:.2f}M")
        >>> print(f"Inference: {cost['inference_time_mean']:.2f}ms")
    """
    model.eval()
    model = model.to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Measure inference time
    dummy_input = torch.randn(input_size).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # Timing
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.time()
            _ = model(dummy_input)
            if device.type == 'mps':
                torch.mps.synchronize()
            elif device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.time()
            times.append((end - start) * 1000)  # Convert to ms

    mean_time = np.mean(times)
    std_time = np.std(times)

    # Estimate memory (rough approximation)
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers())
    total_memory_bytes = param_memory + buffer_memory
    total_memory_mb = total_memory_bytes / (1024 ** 2)

    return {
        'num_parameters': num_params,
        'num_parameters_m': num_params / 1e6,
        'inference_time_mean': mean_time,
        'inference_time_std': std_time,
        'memory_mb': total_memory_mb
    }


def print_model_summary(
    model: torch.nn.Module,
    model_name: str,
    test_metrics: Dict[str, any],
    computational_cost: Dict[str, any]
):
    """
    Print a comprehensive model summary

    Args:
        model: PyTorch model
        model_name: Name of the model
        test_metrics: Metrics from compute_metrics()
        computational_cost: Metrics from compute_computational_cost()

    Example:
        >>> metrics = compute_metrics(predictions)
        >>> cost = compute_computational_cost(model)
        >>> print_model_summary(model, "VanillaCNN", metrics, cost)
    """
    print("\n" + "=" * 70)
    print(f"MODEL SUMMARY: {model_name}")
    print("=" * 70)

    print(f"\n PERFORMANCE METRICS")
    print(f"   Pitch Accuracy:  {test_metrics['pitch']['accuracy']:.4f}")
    print(f"   Pitch Precision: {test_metrics['pitch']['precision']:.4f}")
    print(f"   Pitch Recall:    {test_metrics['pitch']['recall']:.4f}")
    print(f"   Pitch F1:        {test_metrics['pitch']['f1_score']:.4f}")

    print(f"\n   Roll Accuracy:   {test_metrics['roll']['accuracy']:.4f}")
    print(f"   Roll Precision:  {test_metrics['roll']['precision']:.4f}")
    print(f"   Roll Recall:     {test_metrics['roll']['recall']:.4f}")
    print(f"   Roll F1:         {test_metrics['roll']['f1_score']:.4f}")

    avg_acc = (test_metrics['pitch']['accuracy'] + test_metrics['roll']['accuracy']) / 2
    print(f"\n   Average Accuracy: {avg_acc:.4f}")

    print(f"\n💻 COMPUTATIONAL COST")
    print(f"   Parameters:      {computational_cost['num_parameters_m']:.2f}M")
    print(f"   Inference Time:  {computational_cost['inference_time_mean']:.2f} ± {computational_cost['inference_time_std']:.2f} ms")
    print(f"   Memory:          {computational_cost['memory_mb']:.2f} MB")

    print("\n" + "=" * 70)
