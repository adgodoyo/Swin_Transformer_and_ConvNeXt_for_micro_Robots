# ConvNeXt and Swin Transformer for Microrobot Pose and Depth Estimation

**Deep Learning Assignment | Imperial College London**

A comprehensive dual-task deep learning project for surgical robot pose estimation from microscope images, featuring both **pose classification** (pitch + roll angles) and **depth regression** with state-of-the-art architectures and transfer learning techniques. ConvNeXt and Swin Transformers—have introduced hierarchical designs that combine convolutional inductive biases with transformer-inspired components, and have shown state-of-the-art performance on natural-image benchmarks.

---
## In short: Testing models

For testing the models, go to the notebook `0__test_all_models.ipynb`, run the setup and initial imports, and set the desired data path.

Due to file size limitations, some models are not included in the GitHub repository, so you will need to run the corresponding notebooks to generate those files.

---

## 📁 Complete Project Structure

```
Project/
├── data/                             # Original dataset (40 pose configs, ~2000 images)
│   ├── P0_R0/                        # Pose: pitch=0°, roll=0° (50+ images + depth.txt)
│   ├── P15_R30/                      # Pose: pitch=15°, roll=30°
│   └── ... (40 total pose folders)   # Not all 18×7=126 combinations exist
│
├── notebooks/                         # 8 notebooks total (ALL COMPLETED)
│   ├── 0__test_all_models.ipynb      # Testing harness for all 10 models. # THE TEST FILE
│   ├── 0_comprehensive_eda.ipynb     # Complete EDA with manifold analysis
│   ├── 1_vanilla_cnn.ipynb           # Pose classification (pitch+roll)
│   ├── 2_convnext_v2.ipynb           # Pose classification (pretrained)
│   ├── 3_swin_transformer_v2.ipynb   # Pose classification (pretrained)
│   ├── 4_vanilla_cnn_depth.ipynb     # Depth regression
│   ├── 5_convnext_v2_depth.ipynb     # Depth regression (transfer learning)
│   └── 6_swin_v2_depth.ipynb         # Depth regression (2-phase training)

│
├── src/
│   ├── data/
│   │   ├── pose_dataset.py           # Classification dataset (pitch + roll)
│   │   └── depth_dataset.py          # Regression dataset (depth values)
│   │
│   └── utils/                        # Complete utilities module (2,551 lines)
│       ├── __init__.py               # Exports all classification & regression utilities
│       ├── training_utils.py         # Classification training loops (392 lines)
│       ├── data_utils.py             # Data splitting & loaders (183 lines)
│       ├── visualization_utils.py    # Classification visualizations (526 lines)
│       ├── evaluation_utils.py       # Classification metrics (350 lines)
│       ├── regression_utils.py       # Regression utilities (929 lines)
│       └── swin_final_attention.py   # Swin attention viz (109 lines)
│
├── models/                            # 10 trained models (~1.4GB total)
│   ├── vanilla_cnn_baseline_best.pth          # Classification (baseline)
│   ├── vanilla_cnn_optimized_best.pth         # Classification (Optuna optimized)
│   ├── convnextv2_baseline_best.pth           # Classification (pretrained)
│   ├── swin_v2_baseline_best.pth              # Classification (pretrained)
│   ├── 7_vanilla_cnn_depth_baseline.pth       # Regression (baseline)
│   ├── vanilla_cnn_optimized_depth_best.pth   # Regression (optimized)
│   ├── 8_convnext_v2_depth_backboneFR.pth     # Regression (frozen backbone)
│   ├── 8_convnext_v2_depth_Unfrozen.pth       # Regression (unfrozen backbone)
│   ├── 9_swin_v2_depth_phase1.pth             # Regression (phase 1: frozen)
│   └── 9_swin_v2_depth_phase2.pth             # Regression (phase 2: fine-tuned)
│
└── docs/
    ├── 0_Latex_Report.pdf             # Report                             
    └── UTILITIES_QUICK_REFERENCE.md   # Quick reference guide
```

---

## 🏗️ Project Architecture

### 1. Dataset

**Source:** Surgical robot microscope images (40 pose configurations - Only robot 3), released by Lan Wei and Dandan Zhang in [A Dataset and Benchmarks for Deep Learning-Based Optical Microrobot Pose and Depth Perception](https://arxiv.org/abs/2505.18303). Dataset available [here](https://huggingface.co/datasets/Lan-2025/OpticalMicrorobot/tree/main/Deep_Learning_2025_Dataset).

- **Total images:** ~2,000
- **Pose configurations:** 40 (subset of 18 pitch × 7 roll combinations)
- **Images per pose:** ~50 at varying depths 
- **Splits:** 60% train / 20% val / 20% test (stratified)

**Data Structure:**
```
data/
├── P0_R0/          # Pitch=0°, Roll=0°
│   ├── frame_0.jpg
│   ├── frame_1.jpg
│   ├── ...
│   └── depth.txt   # Depth values for each frame
├── P15_R30/        # Pitch=15°, Roll=30°
└── ...             # 40 total pose folders
```

---

### **2. Utilities Module** (2,551 lines total)

**Datasets:**
- `PoseDataset` (pose_dataset.py): Returns (image, pitch_label, roll_label)
- `RegressDataset` (depth_dataset.py): Returns (image, depth_value)

#### **Classification Utilities** (1,451 lines)
Comprehensive suite for dual-head pose classification:

**data_utils.py** (183 lines):
- `create_stratified_splits()` - 60-20-20 stratified split maintaining class distribution
- `create_dataloaders()` - Create train/val/test loaders with MPS optimization
- `get_class_mappings()` - Extract class names and mappings

**training_utils.py** (392 lines):
- `MetricsTracker` - Track metrics across epochs
- `EarlyStopping` - Prevent overfitting
- `train_epoch()` - Train one epoch with dual heads
- `validate_epoch()` - Validate one epoch
- `collect_predictions()` - Collect predictions for analysis


**visualization_utils.py** (526 lines):
- `plot_training_curves()` - 6-panel training dynamics
- `plot_confusion_matrices()` - Dual confusion matrices (pitch + roll)
- `plot_per_class_performance()` - Precision/recall/F1 bars
- `plot_optuna_results()` - Hyperparameter optimization analysis
- `plot_feature_maps()` - Conv layer activations
- `plot_grad_cam()` - Attention heatmaps
- `plot_embedding_tsne()` - Feature space visualization (2D/3D)

**evaluation_utils.py** (350 lines):
- `compute_metrics()` - Accuracy, precision, recall, F1 for both heads
- `generate_classification_report()` - Detailed per-class reports
- `analyze_errors()` - Error patterns and confusion analysis
- `compute_computational_cost()` - Parameters, FLOPs, inference time
- `print_model_summary()` - Comprehensive summary table

#### **Regression Utilities** (929 lines)
Complete suite for depth regression:

**regression_utils.py** (929 lines):

**Training Functions:**
- `train_epoch_regression()` - Train one epoch for regression
- `validate_epoch_regression()` - Validate one epoch for regression
- `collect_predictions_regression()` - Collect predictions with images
- `RegressionMetricsTracker` - Track regression metrics over epochs

**Metrics Functions:**
- `compute_regression_metrics()` - RMSE, MAE, R², Median AE, Max Error, Std Error

**Visualization Functions:**
- `plot_regression_training_curves()` - Loss/RMSE/MAE over epochs (5 panels)
- `plot_predictions_vs_actual()` - Scatter plot with perfect prediction line
- `plot_residuals()` - Residual analysis (3 panels)
- `plot_error_distribution()` - Error histograms and Q-Q plots
- `plot_depth_predictions_gallery()` - Visual gallery of predictions

**Transfer Learning Functions:**
- `load_classification_backbone()` - Load pretrained classification model
- `freeze_backbone()` - Freeze backbone layers for transfer learning
- `unfreeze_backbone()` - Unfreeze layers for fine-tuning

#### **Specialized Utilities** (109 lines)
**swin_final_attention.py**:
- Attention map visualization for Swin Transformers
- Window-based attention analysis

---

### **3. Model Architectures**

#### **Classification Models (4 total)**

**1. VanillaCNN Baseline**
- 5-layer CNN with dual heads
- Channels: [32, 64, 128, 256, 512]
- FC dimension: 256, Dropout: 0.5
- ~1.7M parameters
- **Performance:** ~99% average accuracy

**2. VanillaCNN Optimized**
- Optuna-tuned architecture
- Channels: [48, 32, 128, 128, 64] (more efficient!)
- FC dimension: 384, Dropout: 0.3
- ~656K parameters (62% reduction!)
- **Performance:** ~99.5% average accuracy

**3. ConvNeXt V2 Baseline**
- Pretrained on ImageNet-1K
- Modern ConvNet architecture
- ~28M parameters
- **Performance:** ~99.5% average accuracy

**4. Swin Transformer V2 Baseline**
- Pretrained on ImageNet-1K
- Window-based attention (window16)
- ~28M parameters
- Input: 256×256
- **Performance:** ~99.5% average accuracy

#### **Regression Models (6 total)**

**5. VanillaCNN Depth Baseline**
- Single regression head
- Channels: [32, 64, 128, 256, 512]
- FC dimension: 256
- **Performance:** RMSE: 0.0837, R²: 0.9301

**6. VanillaCNN Depth Optimized**
- Architecture from classification tuning
- Channels: [48, 32, 128, 128, 64]
- FC dimension: 384
- **Performance:** RMSE: 0.0605, R²: 0.9635

**7. ConvNeXt V2 Depth (Frozen Backbone)**
- Transfer learning from classification
- Backbone frozen, only regression head trained
- **Performance:** RMSE: 0.0420, R²: 0.9824

**8. ConvNeXt V2 Depth (Unfrozen / Full Fine-tuning)**
- Full model fine-tuning
- All layers trainable
- **Performance:** RMSE: 0.0417, R²: 0.9826

**9. Swin V2 Depth Phase 1 (Frozen)**
- Transfer learning with frozen backbone
- Only regression head trained
- **Performance:** RMSE: 0.0772, R²: 0.9406

**10. Swin V2 Depth Phase 2 (Fine-tuned)**
- 2-phase training: freeze → unfreeze
- Full model fine-tuning after phase 1
- **Performance:** RMSE: 0.0504, R²: 0.9747

---

## Key Insights & Achievements

### **1. Architecture Insights:**
- **VanillaCNN Optimized:** 62% parameter reduction with better accuracy
- **Transfer Learning:** Massive performance boost (RMSE: 0.084 → 0.042)
- **2-Phase Training:** Significant improvement over single-phase (RMSE: 0.077 → 0.050)

### **2. Transfer Learning Benefits:**
- Classification → Regression transfer works extremely well
- Frozen backbone gives 98% of unfrozen performance with faster training
- 2-phase training provides best balance of speed and performance

### **3. Model Comparison:**
**Classification:**
- All models achieve >99% accuracy
- VanillaCNN Optimized: Best efficiency (656K params, 99.5% acc)
- ConvNeXt V2 & Swin V2: Slightly better but 40× more parameters

**Regression:**
- ConvNeXt V2 Unfrozen: Best overall (RMSE: 0.0417, R²: 0.9826)
- ConvNeXt V2 Frozen: Best efficiency (similar performance, faster training)
- VanillaCNN: Good baseline but transfer learning dominates

### **4. Computational Efficiency:**
- VanillaCNN: ~1-2ms inference (fastest)
- ConvNeXt V2: ~12-15ms inference
- Swin V2: ~23-25ms inference (slowest but accurate)

---

## Hardware 

**Applied Optimizations:**
- `num_workers=0` in DataLoaders (MPS works best single-threaded)
- `pin_memory=False` in DataLoaders (unified memory architecture)
- Mixed precision support for Transformer models
- Gradient checkpointing for memory-intensive models
- Batch size tuning per architecture

**Actual Performance:** (System: Apple M4 + 24GB Unified Memory)
- VanillaCNN: ~1.6ms inference, 617 FPS
- ConvNeXt V2: ~12.4ms inference, 80 FPS
- Swin V2: ~24.0ms inference, 42 FPS

---

*Last updated: 2025-12-13*
*Status: **PROJECT COMPLETE** - All models trained, documented, and tested*
*Total development time: ~1 week*
*Code quality: Production-ready with comprehensive documentation*
