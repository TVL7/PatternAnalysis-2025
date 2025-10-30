import os, time, argparse, torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import build_datasets, build_loaders, set_seed
from modules import (
    build_model, get_loss, train_epoch, evaluate,
    youden_threshold, best_f1_threshold, eval_with_threshold,
    save_history, plot_loss_curves, plot_val_scores, plot_extras, plot_roc_cm
)

def parse_args():
    p = argparse.ArgumentParser(description="Train ConvNeXt-Tiny on AD vs NC")
    p.add_argument("--root", type=str, default="./AD_NC", help="Dataset root containing train/ and test/")
    p.add_argument("--out",  type=str, default="./outputs", help="Output directory")
    p.add_argument("--im_size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch",  type=int, default=32)
    p.add_argument("--lr",     type=float, default=1e-4)
    p.add_argument("--wd",     type=float, default=1e-4)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--class_weights", type=float, nargs=2, default=[1.0, 3.0], help="[NC, AD] order")
    p.add_argument("--val_split", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()