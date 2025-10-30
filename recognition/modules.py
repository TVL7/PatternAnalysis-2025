

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
