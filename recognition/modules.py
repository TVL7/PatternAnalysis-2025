

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
    return None

