

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
    """
    Expects directory structure:
      root/
        train/AD, train/NC
        test/AD,  test/NC
    """
    set_seed(seed)
    train_dir = os.path.join(root, "train")
    test_dir  = os.path.join(root, "test")

    train_tf, eval_tf = get_transforms(im_size)
    full_train_aug = datasets.ImageFolder(train_dir, transform=train_tf)   # for actual training
    full_train_lbl = datasets.ImageFolder(train_dir, transform=eval_tf)    # for eval/val transform
    test_ds        = datasets.ImageFolder(test_dir,  transform=eval_tf)

    idxs = list(range(len(full_train_aug)))
    random.shuffle(idxs)
    n_val = max(1, int(val_split * len(idxs)))
    val_idxs, train_idxs = idxs[:n_val], idxs[n_val:]
    train_ds = Subset(full_train_aug, train_idxs)
    val_ds   = Subset(full_train_lbl, val_idxs)

    meta = {
        "classes": full_train_lbl.classes,
        "train_len": len(train_ds),
        "val_len": len(val_ds),
        "test_len": len(test_ds),
    }
    return train_ds, val_ds, test_ds, meta

def build_loaders(
    train_ds, val_ds, test_ds,
    batch_size: int, num_workers: int = 2
) -> Tuple[DataLoader, DataLoader, DataLoader]: