

def set_seed(seed: int):
    import numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def get_transforms(im_size: int):

def build_datasets(root: str, im_size: int, val_split: float, seed: int) -> Tuple[Subset, Subset, datasets.ImageFolder, Dict[str, Any]]:   


def build_loaders(
    train_ds, val_ds, test_ds,
    batch_size: int, num_workers: int = 2
) -> Tuple[DataLoader, DataLoader, DataLoader]: