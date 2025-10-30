# Alzheimer's Disease Classification with ConvNeXt-Tiny (AD vs NC)
This project fine-tunes ConvNeXt-Tiny (ImageNet-1K pretrained) to classify MRI slices as Alzheimer’s Disease (AD) vs Normal Control (NC). It includes reproducible data loading, a mixed-precision training loop, validation-based threshold selection (Youden’s J / best-F1), and a small evaluation script.


## 1. Problem & Approach (What it solves, how it works)
**Problem:** Distinguishing AD from NC in MRI is challenging due to subtle anatomical changes and class imbalance.
**Approach:** We replace onvNeXt-Tiny's classifier with `Dropout(0.5) -> Linear(2)` and fine-tune using AdamW, Cosine LR, label smoothing, and class-weighted cross-entropy. During training we track AUROC, F1, and Accuracy, save the best checkpoint by validation AUROC, then choose a decision threshold on validation (Youden / best-F1) before reporting on test.


