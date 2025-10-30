import os, argparse, torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from modules import build_model, get_loss, evaluate, eval_with_threshold

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained ConvNeXt-Tiny checkpoint")
    p.add_argument("--data_dir", type=str, required=True, help="Directory with class subfolders (e.g., AD/ NC/)")
    p.add_argument("--ckpt", type=str, required=True, help="Path to best.pt")