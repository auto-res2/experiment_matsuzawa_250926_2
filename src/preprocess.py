"""src/preprocess.py
Dataset loading, preprocessing and episodic sampling now specialised for the
real miniImageNet benchmark (GATE-engine/mini_imagenet on 🤗 Hub).  Synthetic
SMOKE dataset is still retained for quick pipeline checks.
"""
from __future__ import annotations

import random
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset

# -----------------------------------------------------------------------------
#                              TRANSFORMS
# -----------------------------------------------------------------------------

def _imagenet_stats():
    # Standard ImageNet mean/std for normalisation
    return ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def get_transforms(split: str, cfg: Dict) -> transforms.Compose:
    name = cfg["dataset"]["name"].lower()
    if name == "smoke_dataset":
        # trivial transform for synthetic data
        return transforms.Compose([transforms.ToTensor()])

    elif name == "miniimagenet":
        mean, std = _imagenet_stats()
        if split == "train":
            t = transforms.Compose(
                [
                    transforms.RandomResizedCrop(84),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
        else:
            t = transforms.Compose(
                [
                    transforms.Resize(92),
                    transforms.CenterCrop(84),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
        return t

    else:
        raise ValueError(f"Unknown dataset name '{name}'.")


# -----------------------------------------------------------------------------
#                        HUGGING FACE DATASET WRAPPERS
# -----------------------------------------------------------------------------

class MiniImageNetDataset(Dataset):
    """Wrapper around GATE-engine/mini_imagenet with torchvision-style __getitem__."""

    def __init__(self, split: str, transform: transforms.Compose):
        assert split in {"train", "validation", "test"}, "miniImageNet uses train/validation/test splits"
        ds = load_dataset("GATE-engine/mini_imagenet", split=split)
        self.images = ds["image"]
        self.labels = ds["label"]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        img = self.transform(img)
        label = int(self.labels[idx])
        return img, label


# -----------------------------------------------------------------------------
#                            SYNTHETIC SMOKE DATASET
# -----------------------------------------------------------------------------

class _SyntheticDataset(Dataset):
    """Simple synthetic dataset: coloured noise images with class labels."""

    def __init__(self, cfg: Dict, split: str):
        self.num_classes = cfg["dataset"]["num_classes"]
        self.samples_per_class = cfg["dataset"].get("num_samples_per_class", 20)
        c, h, w = cfg["dataset"]["input_shape"]
        self.X = torch.randn(self.num_classes * self.samples_per_class, c, h, w)
        self.y = torch.arange(self.num_classes).repeat_interleave(self.samples_per_class)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], int(self.y[idx])


# -----------------------------------------------------------------------------
#                        EPISODIC SAMPLER & DATALOADER
# -----------------------------------------------------------------------------

class EpisodeDataset(Dataset):
    """Creates on-the-fly episodes (support + query) from a base dataset."""

    def __init__(self, base_dataset: Dataset, cfg: Dict, split: str):
        self.base_dataset = base_dataset
        self.cfg = cfg
        self.split = split
        # build index per class
        self.cls_to_indices: Dict[int, List[int]] = {}
        for idx in range(len(base_dataset)):
            _, label = base_dataset[idx]
            self.cls_to_indices.setdefault(int(label), []).append(idx)

    def __len__(self):
        if self.split == "train":
            return self.cfg["training"]["episodes_per_epoch"]
        else:
            return self.cfg["evaluation"]["episodes"]

    def __getitem__(self, idx):  # idx unused, episodes random
        way = self.cfg["algo"]["way"]
        shot = self.cfg["algo"]["shot"]
        query = self.cfg["algo"]["query"]
        selected_classes = random.sample(list(self.cls_to_indices.keys()), way)
        imgs: List[torch.Tensor] = []
        labels: List[int] = []
        for new_lbl, cls in enumerate(selected_classes):
            indices = random.sample(self.cls_to_indices[cls], shot + query)
            for orig_idx in indices:
                img, _ = self.base_dataset[orig_idx]
                imgs.append(img)
                labels.append(new_lbl)
        imgs = torch.stack(imgs, dim=0)
        labels = torch.tensor(labels)
        return imgs, labels


# -----------------------------------------------------------------------------
#                       DATALOADER FACTORY (UPDATED)
# -----------------------------------------------------------------------------

def get_dataloaders(cfg: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Returns train/val/test DataLoaders of episodes."""
    name = cfg["dataset"]["name"].lower()

    if name == "smoke_dataset":
        base_train = _SyntheticDataset(cfg, split="train")
        base_val = _SyntheticDataset(cfg, split="val")
        base_test = _SyntheticDataset(cfg, split="test")

    elif name == "miniimagenet":
        tf_train = get_transforms("train", cfg)
        tf_val = get_transforms("validation", cfg)
        base_train = MiniImageNetDataset("train", tf_train)
        base_val = MiniImageNetDataset("validation", tf_val)
        base_test = MiniImageNetDataset("test", tf_val)

    else:
        raise ValueError(f"Unknown dataset '{cfg['dataset']['name']}'")

    train_set = EpisodeDataset(base_train, cfg, split="train")
    val_set = EpisodeDataset(base_val, cfg, split="validation")
    test_set = EpisodeDataset(base_test, cfg, split="test")

    train_loader = DataLoader(
        train_set,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["dataset"].get("num_workers", 4),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg["evaluation"].get("batch_size", 1),
        shuffle=False,
        num_workers=cfg["dataset"].get("num_workers", 4),
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg["evaluation"].get("batch_size", 1),
        shuffle=False,
        num_workers=cfg["dataset"].get("num_workers", 4),
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader
