# Utilities Quick Reference
# Referencia Rápida de Utilidades

---

## 🚀 Quick Start

### Classification Imports

```python
# Import classification utilities
from src.utils import (
    # Training
    MetricsTracker, train_epoch, validate_epoch, EarlyStopping,
    # Data
    create_stratified_splits, create_dataloaders, get_class_mappings,
    # Visualization
    plot_training_curves, plot_confusion_matrices,
    # Evaluation
    compute_metrics, compute_computational_cost, print_model_summary
)
```

### Regression Imports

```python
# Import regression utilities
from src.utils import (
    # Training
    RegressionMetricsTracker, train_epoch_regression, validate_epoch_regression,
    # Metrics
    compute_regression_metrics, collect_predictions_regression,
    # Visualization
    plot_regression_training_curves, plot_predictions_vs_actual,
    plot_residuals, plot_error_distribution, plot_depth_predictions_gallery,
    # Transfer Learning
    load_classification_backbone, freeze_backbone, unfreeze_backbone
)
```

---

## 📊 Common Workflows

### Classification Workflows

#### 1. Setup Data Pipeline (5 lines)

```python
train_idx, val_idx, test_idx = create_stratified_splits(dataset)
train_loader, val_loader, test_loader = create_dataloaders(
    train_ds, val_ds, test_ds, train_idx, val_idx, test_idx, batch_size=64
)
mappings = get_class_mappings(dataset)
```

#### 2. Training Loop (10 lines)

```python
tracker = MetricsTracker()
early_stop = EarlyStopping(patience=10, mode='max', save_path='best.pth')

for epoch in range(100):
    train_metrics = train_epoch(model, train_loader, criterion_p, criterion_r, optimizer, device)
    val_metrics = validate_epoch(model, val_loader, criterion_p, criterion_r, device)
    tracker.update({**{f'train_{k}': v for k, v in train_metrics.items()},
                    **{f'val_{k}': v for k, v in val_metrics.items()},
                    'lr': optimizer.param_groups[0]['lr']})
    if early_stop((val_metrics['pitch_acc'] + val_metrics['roll_acc'])/2, model):
        break
```

#### 3. Evaluation & Visualization (5 lines)

```python
predictions = collect_predictions(model, test_loader, device)
metrics = compute_metrics(predictions)
cost = compute_computational_cost(model, device=device)
plot_training_curves(tracker.get_history(), save_path='curves.png')
plot_confusion_matrices(predictions, pitch_names, roll_names, save_path='cm.png')
```

---

### Regression Workflows

#### 1. Regression Training (From Scratch) - 10 lines

```python
tracker = RegressionMetricsTracker()
criterion = nn.MSELoss()

for epoch in range(50):
    train_metrics = train_epoch_regression(model, train_loader, criterion, optimizer, device)
    val_metrics = validate_epoch_regression(model, val_loader, criterion, device)
    tracker.update({'train_loss': train_metrics['loss'], 'train_rmse': train_metrics['rmse'],
                    'val_loss': val_metrics['loss'], 'val_rmse': val_metrics['rmse'],
                    'lr': optimizer.param_groups[0]['lr']})
    if val_metrics['rmse'] < best_rmse:
        best_rmse = val_metrics['rmse']
        torch.save(model.state_dict(), 'best.pth')
```

#### 2. Regression Evaluation (5 lines)

```python
results = collect_predictions_regression(model, test_loader, device, return_images=True)
metrics = compute_regression_metrics(results['predictions'], results['targets'])
print(f"RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, R²: {metrics['r2']:.4f}")
plot_regression_training_curves(tracker.get_history(), save_path='curves.png')
plot_predictions_vs_actual(results['predictions'], results['targets'], save_path='preds.png')
```

#### 3. Transfer Learning (2-Phase) - 15 lines

```python
# Phase 1: Frozen backbone
backbone, config = load_classification_backbone('model.pth', device)
model = build_regression_model(backbone, config).to(device)
freeze_backbone(model, freeze_layers=['features'])
optimizer_p1 = torch.optim.AdamW(model.parameters(), lr=1e-3)

for epoch in range(20):  # Train regression head only
    train_metrics = train_epoch_regression(model, train_loader, criterion, optimizer_p1, device)
    val_metrics = validate_epoch_regression(model, val_loader, criterion, device)

# Phase 2: Fine-tuning
unfreeze_backbone(model)
optimizer_p2 = torch.optim.AdamW(model.parameters(), lr=1e-4)  # Lower LR!

for epoch in range(30):  # Fine-tune entire model
    train_metrics = train_epoch_regression(model, train_loader, criterion, optimizer_p2, device)
    val_metrics = validate_epoch_regression(model, val_loader, criterion, device)
```

---

## 🔧 Function Quick Ref

### Classification Functions

| Function | Purpose | Returns |
|----------|---------|---------|
| `create_stratified_splits()` | 60-20-20 split | train_idx, val_idx, test_idx |
| `create_dataloaders()` | Make DataLoaders | train_loader, val_loader, test_loader |
| `get_class_mappings()` | Extract class info | Dict with mappings & counts |
| `train_epoch()` | Train 1 epoch (dual-head) | Dict with loss & accuracy |
| `validate_epoch()` | Validate 1 epoch (dual-head) | Dict with loss & accuracy |
| `collect_predictions()` | Get all predictions | Dict with preds, labels, probs |
| `compute_metrics()` | Calculate metrics | Dict with acc, precision, recall, F1 |
| `compute_computational_cost()` | Measure efficiency | Dict with params, time, memory |

### Regression Functions

| Function | Purpose | Returns |
|----------|---------|---------|
| `train_epoch_regression()` | Train 1 epoch (regression) | Dict with loss, RMSE, MAE |
| `validate_epoch_regression()` | Validate 1 epoch (regression) | Dict with loss, RMSE, MAE |
| `collect_predictions_regression()` | Get predictions + images | Dict with predictions, targets, images |
| `compute_regression_metrics()` | Calculate regression metrics | Dict with RMSE, MAE, R², etc. |
| `load_classification_backbone()` | Load pretrained backbone | backbone, config |
| `freeze_backbone()` | Freeze layers | None (modifies in-place) |
| `unfreeze_backbone()` | Unfreeze all layers | None (modifies in-place) |

---

## 🎨 Visualization Quick Ref

### Classification Visualizations

| Function | What It Plots |
|----------|---------------|
| `plot_training_curves()` | 6-panel training dynamics |
| `plot_confusion_matrices()` | Side-by-side confusion matrices |
| `plot_per_class_performance()` | Bar charts of precision/recall/F1 |
| `plot_optuna_results()` | Optimization history & importances |
| `plot_feature_maps()` | Conv layer activations |
| `plot_grad_cam()` | Attention heatmaps |
| `plot_embedding_tsne()` | t-SNE of learned features |

### Regression Visualizations

| Function | What It Plots |
|----------|---------------|
| `plot_regression_training_curves()` | 5-panel training dynamics (MSE, RMSE, MAE, LR) |
| `plot_predictions_vs_actual()` | Scatter plot with perfect line |
| `plot_residuals()` | 3-panel residual analysis |
| `plot_error_distribution()` | 4-panel error distribution (histogram, Q-Q, etc.) |
| `plot_depth_predictions_gallery()` | Visual gallery of predictions |

---

## ⚙️ Configuration Templates

### MPS (Apple Silicon)

```python
device = torch.device('mps')
batch_size = 64
num_workers = 0
pin_memory = False
```

### Mixed Precision

```python
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()
train_metrics = train_epoch(model, loader, criterion_p, criterion_r, optimizer, device, scaler=scaler)
```

### Optuna Objective

```python
def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 5e-2, log=True)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    model = build_model(lr, dropout)
    return train_and_validate(model)
```

---

## 📝 Class Names Helper

```python
mappings = get_class_mappings(dataset)
pitch_names = [mappings['pitch_inv'][i] for i in range(mappings['num_pitch_classes'])]
roll_names = [mappings['roll_inv'][i] for i in range(mappings['num_roll_classes'])]
```

---

## 🐛 Common Errors

| Error | Solution |
|-------|----------|
| MPS OOM | Reduce batch_size |
| KeyError in tracker | Check metric key names match (train_loss, val_loss, etc.) |
| Optuna trial fails | Add try-except with return 0.0 |
| num_workers warning | Set to 0 for MPS |
| NaN in regression | Check for division by zero, use gradient clipping |
| Transfer learning mismatch | Use strict=False in load_classification_backbone() |

---

## 📦 Complete Import Statement

```python
# Import ALL utilities (classification + regression)
from src.utils import *
```

---

**For detailed documentation, see: `docs/UTILITIES_DOCUMENTATION.md`**

*Last updated: 2025-12-12*
