

def build_model(dropout: float = 0.5, num_classes: int = 2, device: str = "cpu"):
    return None

def get_loss(class_weights: torch.Tensor, label_smoothing: float, device: str):
    return None

def train_epoch(model, loader, criterion, opt, scaler: GradScaler, device: str) -> float:
    return None

def evaluate(model, loader, criterion, device: str) -> Dict[str, Any]:
    return None

