

def build_model(dropout: float = 0.5, num_classes: int = 2, device: str = "cpu"):
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    model = convnext_tiny(weights=weights).to(device)

    # Replace head: norm, flatten, dropout, linear
    norm = model.classifier[0]
    flatten = model.classifier[1]
    in_features = model.classifier[2].in_features
    model.classifier = nn.Sequential(
        norm,
        flatten,
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes)
    ).to(device)
    return model

def get_loss(class_weights: torch.Tensor, label_smoothing: float, device: str):
    class_weights = class_weights.to(device) if class_weights is not None else None
    return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

def train_epoch(model, loader, criterion, opt, scaler: GradScaler, device: str) -> float:
    model.train()
    tot = n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        with autocast(enabled=torch.cuda.is_available()):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        bs = x.size(0)
        tot += loss.item() * bs
        n += bs
    return tot / max(n, 1)

def evaluate(model, loader, criterion, device: str) -> Dict[str, Any]:
    model.eval()
    acc  = BinaryAccuracy().to(device)
    auro = BinaryAUROC().to(device)
    f1m  = BinaryF1Score().to(device)
    tot = n = 0
    probs_all, ys_all = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with autocast(enabled=torch.cuda.is_available()):
            logits = model(x)
            loss = criterion(logits, y)
        bs = x.size(0)
        tot += loss.item() * bs; n += bs
        p = torch.softmax(logits, dim=1)[:, 1]
        acc.update(p, y.int()); auro.update(p, y.int()); f1m.update(p, y.int())
        probs_all.append(p.detach().cpu()); ys_all.append(y.detach().cpu())
    return {
        "loss":  tot / max(n, 1),
        "acc":   acc.compute().item(),
        "auroc": auro.compute().item(),
        "f1":    f1m.compute().item(),
        "probs": torch.cat(probs_all).numpy(),
        "ys":    torch.cat(ys_all).numpy(),
    }


def youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    idx = np.argmax(tpr - fpr)
    return float(thr[idx])

def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    ts = np.linspace(0.01, 0.99, 99)
    scores = [f1_score(y_true, (y_prob >= t).astype(int)) for t in ts]
    return float(ts[int(np.argmax(scores))])

def eval_with_threshold(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> Tuple[float, float, np.ndarray]:
    y_pred = (y_prob >= thr).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred)
    cm  = confusion_matrix(y_true, y_pred)
    return acc, f1, cm

def save_history(hist: Dict[str, Any], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(hist, f)


def plot_loss_curves(hist: Dict[str, Any], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    epochs = hist["epoch"]
    # Loss
    plt.figure()
    plt.plot(epochs, hist["train_loss"], label="train")
    plt.plot(epochs, hist["val_loss"], label="val")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.title("Loss")
    plt.tight_layout()
    path = os.path.join(out_dir, "loss_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path

def plot_val_scores(hist: Dict[str, Any], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    epochs = hist["epoch"]
    plt.figure()
    plt.plot(epochs, hist["val_acc"], label="acc")
    plt.plot(epochs, hist["val_f1"], label="f1")
    plt.plot(epochs, hist["val_auroc"], label="auroc")
    plt.xlabel("epoch"); plt.ylabel("score"); plt.legend(); plt.title("Validation Metrics")
    plt.tight_layout()
    path = os.path.join(out_dir, "val_scores.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path

def plot_extras(hist: Dict[str, Any], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    epochs = hist["epoch"]

    # Validation Accuracy (%)
    plt.figure(figsize=(7,4.5))
    plt.plot(epochs, [100.0*a for a in hist["val_acc"]])
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100); plt.grid(alpha=0.25); plt.tight_layout()
    va_path = os.path.join(out_dir, "val_accuracy.png")
    plt.savefig(va_path, dpi=150); plt.close()

    # Loss Curves (Validation then Training as per your example)
    plt.figure(figsize=(7,4.5))
    plt.plot(epochs, hist["val_loss"],   label="Validation Loss")
    plt.plot(epochs, hist["train_loss"], label="Training Loss")
    plt.title("Training Loss and Validation Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.grid(alpha=0.25); plt.tight_layout()
    lv_path = os.path.join(out_dir, "loss_curves_alt.png")
    plt.savefig(lv_path, dpi=150); plt.close()
    return va_path, lv_path

def plot_roc_cm(y_true: np.ndarray, y_prob: np.ndarray, thr_list, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    AUC = auc(fpr, tpr)
    plt.figure(); plt.plot(fpr, tpr); plt.plot([0,1],[0,1],'--')
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC AUC={AUC:.3f}")
    plt.tight_layout()
    roc_path = os.path.join(out_dir, "roc.png")
    plt.savefig(roc_path, dpi=150); plt.close()

    # Confusion matrices
    for name, thr in thr_list:
        cm = confusion_matrix(y_true, (y_prob >= thr).astype(int))
        plt.figure()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix @ {name} thr={thr:.3f}")
        plt.xlabel("Pred"); plt.ylabel("True")
        plt.tight_layout()
        cm_path = os.path.join(out_dir, f"cm_{name.replace(' ', '_')}.png")
        plt.savefig(cm_path, dpi=150); plt.close()

    return roc_path