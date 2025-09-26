"""src/preprocess.py
Dataset loading & preprocessing pipeline shared by all experiments.
Dataset/model specifics are swapped in next phases; generic components
are fully functional.  A synthetic dataset is provided to enable smoke
tests without any external data downloads.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

__all__ = [
    "get_dataloaders",
]


class SyntheticDataset(Dataset):
    """Creates a small random image-classification dataset for pipeline tests."""

    def __init__(self, length: int = 256, num_classes: int = 10, seed: int = 42):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.data = torch.randn(length, 3, 32, 32, generator=g)
        self.targets = torch.randint(0, num_classes, (length,), generator=g)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def get_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """Returns (train_loader, val_loader, test_loader, num_classes).

    *PLACEHOLDER*: If *config['dataset']['name']* is not "synthetic", this
    function should be extended in the next code-generation step to load the
    actual dataset.
    """
    dataset_cfg = config["dataset"]
    name: str = dataset_cfg["name"].lower()
    batch_size: int = int(dataset_cfg.get("batch_size", 32))
    num_workers: int = int(dataset_cfg.get("num_workers", 0))
    seed: int = int(config["training"].get("seed", 42))

    if name == "synthetic":
        full_ds = SyntheticDataset(length=256, num_classes=10, seed=seed)
        # simple split: 60 % train, 20 % val, 20 % test
        n = len(full_ds)
        train_ds, val_ds, test_ds = torch.utils.data.random_split(
            full_ds,
            [int(0.6 * n), int(0.2 * n), n - int(0.8 * n)],
            generator=torch.Generator().manual_seed(seed),
        )
        num_classes = 10
    else:
        # --------------------------------------------------------------
        # PLACEHOLDER: Will be replaced with specific dataset loading logic.
        # --------------------------------------------------------------
        raise NotImplementedError(
            f"Dataset '{name}' not yet implemented in the common foundation."
        )

    # Basic normalisation transform (identity for synthetic data)
    transform = transforms.Compose([transforms.ToTensor()])

    def _dl(ds):
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=isinstance(ds, torch.utils.data.Subset) and ds.indices[0] == 0,
            num_workers=num_workers,
            pin_memory=True,
        )

    return _dl(train_ds), _dl(val_ds), _dl(test_ds), num_classes
