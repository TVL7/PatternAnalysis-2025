

def set_seed(seed: int):
    import numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def get_transforms(im_size: int):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(im_size, (0.9, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf

def build_datasets(root: str, im_size: int, val_split: float, seed: int) -> Tuple[Subset, Subset, datasets.ImageFolder, Dict[str, Any]]:   


def build_loaders(
    train_ds, val_ds, test_ds,
    batch_size: int, num_workers: int = 2
) -> Tuple[DataLoader, DataLoader, DataLoader]: