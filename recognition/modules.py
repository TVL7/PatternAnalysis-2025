

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
    return None

def train_epoch(model, loader, criterion, opt, scaler: GradScaler, device: str) -> float:
    return None

def evaluate(model, loader, criterion, device: str) -> Dict[str, Any]:
    return None

