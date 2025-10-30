import os, argparse, torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from modules import build_model, get_loss, evaluate, eval_with_threshold

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained ConvNeXt-Tiny checkpoint")
    p.add_argument("--data_dir", type=str, required=True, help="Directory with class subfolders (e.g., AD/ NC/)")
    p.add_argument("--ckpt", type=str, required=True, help="Path to best.pt")
    p.add_argument("--im_size", type=int, default=224)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for class=1 (AD)")
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transform and dataset
    eval_tf = transforms.Compose([
        transforms.Resize((args.im_size, args.im_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
    ])
    ds = datasets.ImageFolder(args.data_dir, transform=eval_tf)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print("Classes:", ds.classes, "Num images:", len(ds))

    # Model & loss (weights not needed for eval metrics; use uniform)
    model = build_model(dropout=0.5, num_classes=2, device=device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    criterion = get_loss(class_weights=None, label_smoothing=0.0, device=str(device))

    # Eval
    out = evaluate(model, loader, criterion, str(device))
    acc_t, f1_t, cm_t = eval_with_threshold(out["ys"], out["probs"], args.threshold)

    print(f"[EVAL @ thr={args.threshold:.3f}] acc={acc_t:.3f}  f1={f1_t:.3f}")
    print("Confusion matrix:\n", cm_t)

if __name__ == "__main__":
    main()