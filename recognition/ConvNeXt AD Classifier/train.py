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

def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Device & seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    print("Device:", device)

    # Data
    train_ds, val_ds, test_ds, meta = build_datasets(args.root, args.im_size, args.val_split, args.seed)
    train_loader, val_loader, test_loader = build_loaders(train_ds, val_ds, test_ds, args.batch, args.num_workers)
    print("Classes:", meta["classes"])
    print("Train/Val/Test:", meta["train_len"], meta["val_len"], meta["test_len"])

    # Model, loss, optim, sched
    model = build_model(dropout=0.5, num_classes=2, device=device)
    criterion = get_loss(torch.tensor(args.class_weights), args.label_smoothing, device=str(device))
    opt   = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    # Warm-up
    _ = next(iter(train_loader))
    print("Warm-up batch OK. Starting training...")

    # Train loop
    hist = {"epoch":[], "train_loss":[], "val_loss":[], "val_acc":[], "val_auroc":[], "val_f1":[]}
    best_auc, best_path = -1.0, os.path.join(args.out, "best.pt")

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr = train_epoch(model, train_loader, criterion, opt, scaler, str(device))
        va = evaluate(model, val_loader, criterion, str(device))
        sched.step()
        dt = time.time() - t0

        hist["epoch"].append(ep)
        hist["train_loss"].append(tr)
        hist["val_loss"].append(va["loss"])
        hist["val_acc"].append(va["acc"])
        hist["val_auroc"].append(va["auroc"])
        hist["val_f1"].append(va["f1"])

        print(f"[epoch {ep:02d} | {dt/60:.1f} min] "
              f"train_loss={tr:.4f}  val_loss={va['loss']:.4f}  "
              f"val_acc={va['acc']:.3f}  val_auroc={va['auroc']:.3f}  val_f1={va['f1']:.3f}")

        if va["auroc"] > best_auc:
            best_auc = va["auroc"]
            torch.save(model.state_dict(), best_path)
            print(f"  ↳ saved best (val_auroc={best_auc:.3f})")
    
    # Save curves & history
    save_history(hist, args.out)
    plot_loss_curves(hist, args.out)
    plot_val_scores(hist, args.out)
    va_png, lv_png = plot_extras(hist, args.out)
    print(f"Saved:\n  {va_png}\n  {lv_png}")

    # Load best & threshold selection on VAL
    model.load_state_dict(torch.load(best_path, map_location=device))
    val_out = evaluate(model, val_loader, criterion, str(device))
    probs_v, ys_v = val_out["probs"], val_out["ys"]

    thr_f1 = best_f1_threshold(ys_v, probs_v)
    thr_y  = youden_threshold(ys_v, probs_v)
    print(f"\n[VAL thresholds] best_F1_thr={thr_f1:.3f}  youden_thr={thr_y:.3f}")
