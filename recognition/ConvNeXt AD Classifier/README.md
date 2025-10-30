# Alzheimer's Disease Classification with ConvNeXt-Tiny (AD vs NC)
**Student Number:** 48906962

**Name:** Tom Van Loon

**Description:** 
This project fine-tunes ConvNeXt-Tiny (ImageNet-1K pretrained) to classify MRI slices as Alzheimer’s Disease (AD) vs Normal Control (NC). It includes reproducible data loading, a mixed-precision training loop, validation-based threshold selection (Youden’s J / best-F1), and a small evaluation script.


## Introduction to ConvNeXt-Tiny

### The Algorithm
ConvNeXt-Tiny is a modern convolutional neural network that “modernizes” a ResNet-style backbone with design cues from Vision Transformers (e.g., larger depthwise kernels, inverted bottlenecks, LayerNorm, and stochastic depth). ConvNeXt keeps a pure-ConvNet hierarchy (patchifying stem → four stages of blocks → global pooling → linear head) and has shown ImageNet-level performance competitive with transformer backbones while remaining efficient and scalable. In this project the model is adapted for binary classification of ADNI MRI slices (Alzheimer’s Disease vs Normal Control) using a class-weighted cross-entropy objective and validation-tuned decision thresholds.

### How it works
Input MR images are resized/normalized to match the ImageNet pretraining statistics, then passed through a patchify stem (4×4, stride 4) to create low-resolution feature maps. The network applies a sequence of ConvNeXt blocks—each block uses a depthwise 7×7 convolution, LayerNorm, a 1×1 expansion (GELU), and a 1×1 projection (with residual connection)—with stage transitions that downsample spatially while increasing channel width. Global average pooling aggregates features; a dropout-regularized linear layer outputs logits for the two classes. During training we optimize with AdamW and a cosine LR schedule; during evaluation we convert logits to probabilities, pick an operating threshold from validation (Youden or best-F1), and report accuracy/F1/AUROC on test.

## Problem & Approach (What it solves, how it works)
**Problem:** Distinguishing AD from NC in MRI is challenging due to subtle anatomical changes and class imbalance.
**Approach:** We replace ConvNeXt-Tiny's classifier with `Dropout(0.5) -> Linear(2)` and fine-tune using AdamW, Cosine LR, label smoothing, and class-weighted cross-entropy. During training we track AUROC, F1, and Accuracy, save the best checkpoint by validation AUROC, then choose a decision threshold on validation (Youden / best-F1) before reporting on test.

## Project Structure
```python
├── dataset.py        # datasets, transforms, val split, dataloaders
├── modules.py        # model/loss builders, train/eval, plotting, thresholds
├── train.py          # training script → best.pt, history.json, figures
├── predict.py        # evaluate a checkpoint on a folder
├── requirements.txt  # pinned versions
└── AD_NC/
    ├── train/AD/   train/NC/
    └── test/AD/    test/NC/
```

## Pre-Processing & Augmentations
- **Input size / Resize:** `RandomResizeCrop(224, scale=(0.9, 1.0))` keeps most anatomy while adding slight scale variation for robustness
- **Light Geometric Augments:** `RandomHorizontalFlip(p=0.5)` and `RandomRotation(±10°)` provide invariance to small pose differences without distorting anatomy
- **Normalization (ImageNet stats):** `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`. Matching ConvNeXt's pretraining distribution speeds convergence and stabilises fine-tuning
- **Validation Set:** The `test/` set remains untouched for final reporting. A **15% validation split** is sampled from `train/` with a fixed seed to tune hyper-parameters and select the operating threshold while preserving most data for training

## Dependencies
```python
Python: 3.12
torch: 2.4.1
torchvision: 0.19.1
timm: 1.0.9
torchmetrics: 1.4.2
scikit-learn: 1.5.1
matplotlib: 3.9.2
seaborn: 0.13.2
numpy: 1.26.4
```




