"""src/preprocess.py
Common preprocessing & data-loading pipeline – now wired to actual
Hugging-Face corruption benchmarks (CIFAR-C, ImageNet-C).
"""
from __future__ import annotations

import random
from typing import Any, Dict

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

from datasets import load_dataset  # NEW – required for real datasets

__all__ = ["get_dataloader", "set_random_seed"]

# --------------------------------------------------------------------------- #
#                         Reproducibility helpers                              #
# --------------------------------------------------------------------------- #

def set_random_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # slower but deterministic
    torch.backends.cudnn.benchmark = False

# --------------------------------------------------------------------------- #
#                          Dataset wrappers                                    #
# --------------------------------------------------------------------------- #

class HFImageDataset(Dataset):
    """Wrap a 🤗 dataset dict that returns PIL images into a torch Dataset."""

    def __init__(self, hf_id: str, split: str, transform, num_samples: int = 0):
        super().__init__()
        self.ds = load_dataset(hf_id, split=split, trust_remote_code=True)
        if num_samples > 0:
            self.ds = self.ds.select(range(min(num_samples, len(self.ds))))
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        img = sample["image"]  # PIL Image
        label = int(sample["label"])
        img = self.transform(img)
        return img, label

# --------------------------------------------------------------------------- #
#                        Transform  utilities                                  #
# --------------------------------------------------------------------------- #

def _get_transforms(name: str):
    """Return torchvision transforms for the dataset."""
    name = name.lower()
    if name.startswith("cifar"):
        mean = [0.491, 0.482, 0.447]
        std = [0.247, 0.243, 0.262]
        return T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    if name.startswith("imagenet"):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        return T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    # default
    return T.Compose([T.ToTensor()])

# --------------------------------------------------------------------------- #
#                          Dataloader builder                                  #
# --------------------------------------------------------------------------- #

def get_dataloader(cfg: Dict[str, Any], split: str = "test") -> DataLoader:
    """Universal dataloader builder – now with real dataset integration."""
    dataset_name = cfg["dataset"]["name"].upper()
    batch_size = cfg["dataset"].get("batch_size", 64)
    num_workers = int(cfg["dataset"].get("workers", 4))
    num_samples = int(cfg["dataset"].get("num_samples", 0))

    # =====================  REAL DATASETS SECTION  ====================== #
    if dataset_name == "CIFAR10C":
        hf_id = "randall-lab/cifar10-c"
        ds = HFImageDataset(hf_id, split="test", transform=_get_transforms("cifar"), num_samples=num_samples)
    elif dataset_name == "CIFAR100C":
        hf_id = "randall-lab/cifar100-c"
        ds = HFImageDataset(hf_id, split="test", transform=_get_transforms("cifar"), num_samples=num_samples)
    elif dataset_name == "IMAGENETC":
        hf_id = "ang9867/ImageNet-C"
        ds = HFImageDataset(hf_id, split="test", transform=_get_transforms("imagenet"), num_samples=num_samples)
    else:
        raise ValueError(f"Unknown or unsupported dataset: {dataset_name}")
    # =================================================================== #

    shuffle = split == "train"
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return dl
