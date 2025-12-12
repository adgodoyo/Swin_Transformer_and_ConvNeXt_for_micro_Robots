"""
Shared utilities for pose classification and depth regression models
"""

# Classification utilities
from .training_utils import (
    MetricsTracker,
    train_epoch,
    validate_epoch,
    collect_predictions,
    EarlyStopping
)

from .data_utils import (
    create_stratified_splits,
    create_dataloaders,
    get_class_mappings
)

from .visualization_utils import (
    plot_training_curves,
    plot_confusion_matrices,
    plot_per_class_performance,
    plot_optuna_results,
    plot_feature_maps,
    plot_grad_cam,
    plot_embedding_tsne
)

from .evaluation_utils import (
    compute_metrics,
    generate_classification_report,
    analyze_errors,
    compute_computational_cost,
    print_model_summary
)

# Regression utilities
from .regression_utils import (
    # Training
    train_epoch_regression,
    validate_epoch_regression,
    collect_predictions_regression,
    RegressionMetricsTracker,
    # Metrics
    compute_regression_metrics,
    # Visualization
    plot_regression_training_curves,
    plot_predictions_vs_actual,
    plot_residuals,
    plot_error_distribution,
    plot_depth_predictions_gallery,
    # Transfer Learning
    load_classification_backbone,
    freeze_backbone,
    unfreeze_backbone
)

__all__ = [
    # Classification - Training
    'MetricsTracker',
    'train_epoch',
    'validate_epoch',
    'collect_predictions',
    'EarlyStopping',
    # Classification - Data
    'create_stratified_splits',
    'create_dataloaders',
    'get_class_mappings',
    # Classification - Visualization
    'plot_training_curves',
    'plot_confusion_matrices',
    'plot_per_class_performance',
    'plot_optuna_results',
    'plot_feature_maps',
    'plot_grad_cam',
    'plot_embedding_tsne',
    # Classification - Evaluation
    'compute_metrics',
    'generate_classification_report',
    'analyze_errors',
    'compute_computational_cost',
    'print_model_summary',
    # Regression - Training
    'train_epoch_regression',
    'validate_epoch_regression',
    'collect_predictions_regression',
    'RegressionMetricsTracker',
    # Regression - Metrics
    'compute_regression_metrics',
    # Regression - Visualization
    'plot_regression_training_curves',
    'plot_predictions_vs_actual',
    'plot_residuals',
    'plot_error_distribution',
    'plot_depth_predictions_gallery',
    # Regression - Transfer Learning
    'load_classification_backbone',
    'freeze_backbone',
    'unfreeze_backbone'
]
