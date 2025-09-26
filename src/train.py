"""src/train.py
Core training logic implementing ProtoNet and PMR-Proto with complete
infrastructure (optimiser, scheduler, checkpointing, history collection).
Now specialised with real backbones (ResNet-12, ConvNeXt-Tiny) so that the
framework is fully functional for the full experiments.
"""
from __future__ import annotations

import os
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

# -----------------------------------------------------------------------------
#                         BACKBONE DEFINITIONS & HELPERS
# -----------------------------------------------------------------------------

class Conv4(nn.Module):
    """A minimal 4-layer ConvNet for the SMOKE test."""

    def __init__(self, in_channels: int = 3, hidden: int = 64, out_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            self._block(in_channels, hidden),  # 32×32 → 16×16
            self._block(hidden, hidden),       # 16×16 → 8×8
            self._block(hidden, hidden),       # 8×8  → 4×4
            self._block(hidden, out_dim),      # 4×4  → 2×2
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    @staticmethod
    def _block(cin: int, cout: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        feat = self.avg_pool(feat)
        return feat.view(feat.size(0), -1)


# -----------------------------------------------------------------------------
#                 REAL BACKBONES FOR FULL EXPERIMENTS
# -----------------------------------------------------------------------------

class _ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        out = self.relu(out)
        return out


class ResNet12(nn.Module):
    """ResNet-12 backbone widely used in few-shot literature.
    Configuration: 4 stages with {64,160,320,640} channels, each made of a
    single residual block followed by 2×2 max-pool.
    Output: 640-D global-average-pooled embedding.
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()
        channels = [64, 160, 320, 640]
        layers = []
        c_in = in_channels
        for c_out in channels:
            layers.append(_ResidualBlock(c_in, c_out, stride=1))
            layers.append(nn.MaxPool2d(2))  # reduce spatial size
            c_in = c_out
        self.encoder = nn.Sequential(*layers)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = channels[-1]

    def forward(self, x):
        x = self.encoder(x)
        x = self.avg_pool(x)
        return x.view(x.size(0), -1)


# -----------------------------------------------------------------------------
#                        BACKBONE FACTORY (UPDATED)
# -----------------------------------------------------------------------------


def get_backbone(name: str, **kwargs) -> nn.Module:
    """Factory for backbones.

    Supported names (case-insensitive):
    • CONV4_SMOKE  –  4-layer toy ConvNet for pipeline testing.
    • RESNET12     –  64-160-320-640 ResNet-12 used in main experiments.
    • CONVNEXT_TINY – ConvNeXt-Tiny (ImageNet-12k pre-train) from timm with
                       classifier head removed so that it outputs embeddings.
    """
    name_upper = name.upper()

    if name_upper == "CONV4_SMOKE":
        return Conv4(
            in_channels=kwargs.get("in_channels", 3),
            hidden=kwargs.get("hidden", 64),
            out_dim=kwargs.get("embedding_dim", 64),
        )

    elif name_upper == "RESNET12":
        return ResNet12(in_channels=kwargs.get("in_channels", 3))

    elif name_upper == "CONVNEXT_TINY":
        import timm  # lazy import so that smoke-test does not require timm

        model = timm.create_model(
            "convnext_tiny.in12k_ft_in1k",
            pretrained=kwargs.get("pretrained", False),
            num_classes=0,  # remove classifier to obtain embeddings
        )
        return model

    else:
        raise ValueError(f"Unknown backbone '{name}'. Supported: CONV4_SMOKE, RESNET12, CONVNEXT_TINY")


# -----------------------------------------------------------------------------
#                               LOSS  FUNCTIONS
# -----------------------------------------------------------------------------

def episode_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    way: int,
    tau: float = 0.2,
    lam: float = 0.1,
    gam: float = 0.01,
    temperature: float = 10.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute PMR-Proto loss and return per-component statistics."""
    n_support = embeddings.size(0) // 2  # support first, query second
    emb_sup, emb_q = embeddings[:n_support], embeddings[n_support:]
    lbl_sup, lbl_q = labels[:n_support], labels[n_support:]

    # ------------------ standard proto loss ------------------
    protos = []
    for c in range(way):
        protos.append(emb_sup[lbl_sup == c].mean(0))
    protos = torch.stack(protos, dim=0)
    protos = F.normalize(protos, dim=1)

    logits = (
        F.normalize(emb_q, dim=1) @ protos.t() * temperature
    )  # [n_query, way]
    L_proto = F.cross_entropy(logits, lbl_q)

    # ----------------- prototype margin loss -----------------
    cos_mat = protos @ protos.t()  # [way, way]
    tri_mask = torch.triu(torch.ones_like(cos_mat), 1).bool()
    cos_pairs = cos_mat[tri_mask]
    L_margin = F.relu(cos_pairs - tau).mean()

    # --------------- feature-norm variance loss ---------------
    L_norm = emb_sup.norm(dim=1).var()

    total = L_proto + lam * L_margin + gam * L_norm
    stats = {
        "L_proto": L_proto.item(),
        "L_margin": L_margin.item(),
        "L_norm": L_norm.item(),
        "L_total": total.item(),
    }
    return total, stats


# -----------------------------------------------------------------------------
#                               TRAINING  LOOP
# -----------------------------------------------------------------------------

def _step(
    model: nn.Module,
    optimiser: torch.optim.Optimizer,
    episode: Tuple[torch.Tensor, torch.Tensor],
    cfg_algo: Dict,
    device: torch.device,
):
    model.train()
    imgs, labels = episode
    imgs = imgs.to(device)
    labels = labels.to(device)
    embeddings = model(imgs)

    loss, _ = episode_loss(
        embeddings,
        labels,
        way=cfg_algo["way"],
        tau=cfg_algo["tau"],
        lam=cfg_algo["lambda"],
        gam=cfg_algo["gamma"],
        temperature=cfg_algo["temperature"],
    )
    optimiser.zero_grad(set_to_none=True)
    loss.backward()
    if cfg_algo.get("grad_clip", None):
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_algo["grad_clip"])
    optimiser.step()

    return loss.item()


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Dict,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """Meta-training loop with checkpointing.
    Returns the best model (in eval mode) and history dict.
    """
    epochs = cfg["training"]["epochs"]
    chkpt_dir = Path(cfg["training"]["checkpoint_dir"])
    chkpt_dir.mkdir(parents=True, exist_ok=True)

    optimiser = SGD(
        model.parameters(),
        lr=cfg["training"]["lr"],
        momentum=0.9,
        weight_decay=cfg["training"].get("weight_decay", 0.0),
        nesterov=True,
    )

    if cfg["training"].get("lr_scheduler", "none").lower() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, epochs)
    else:
        scheduler = None

    history = {"train_loss": [], "val_acc": []}
    best_acc = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        epoch_losses = []
        with tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}") as tloader:
            for episode in tloader:
                loss_val = _step(
                    model,
                    optimiser,
                    episode,
                    cfg["algo"],
                    device,
                )
                epoch_losses.append(loss_val)
                tloader.set_postfix(loss=np.mean(epoch_losses))
        history["train_loss"].append(np.mean(epoch_losses))

        # ---------- validation ----------
        val_metrics = validate(model, val_loader, cfg, device)
        history["val_acc"].append(val_metrics["mean_accuracy"])

        if val_metrics["mean_accuracy"] > best_acc:
            best_acc = val_metrics["mean_accuracy"]
            best_state = {
                "model": model.state_dict(),
                "acc": best_acc,
                "epoch": epoch,
                "cfg": cfg,
            }
            torch.save(best_state, chkpt_dir / "best_model.pt")

        if scheduler is not None:
            scheduler.step()

    # load best state for downstream evaluation
    model.load_state_dict(best_state["model"])
    model.eval()
    return model, history


# -----------------------------------------------------------------------------
#                        VALIDATION (SHARED WITH EVALUATION)
# -----------------------------------------------------------------------------

def validate(
    model: nn.Module,
    val_loader: DataLoader,
    cfg: Dict,
    device: torch.device,
) -> Dict[str, float]:
    from .evaluate import evaluate  # local import to avoid circular dep

    model.eval()
    return evaluate(model, val_loader, cfg, device, disable_figures=True)
