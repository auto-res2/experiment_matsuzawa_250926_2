"""src/evaluate.py
Common evaluation utilities with identical metrics and
result-formatting across all experimental variations.
"""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn import metrics as skm
from torch import nn
from torch.utils.data import DataLoader

__all__ = [
    "evaluate_model",
    "generate_figures",
]

sns.set_theme(style="whitegrid")


@torch.no_grad()
def _predict(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_probs: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    for batch in loader:
        inputs, targets = batch
        inputs = inputs.to(device, non_blocking=True)
        outputs = model(inputs)
        probs = outputs.softmax(dim=1).cpu()
        all_probs.append(probs)
        all_labels.append(targets.cpu())

    probs_tensor = torch.cat(all_probs, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)
    return probs_tensor, labels_tensor


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> Dict:
    """Runs model on *loader* and returns standardised metric dict."""
    probs, labels = _predict(model, loader, device)
    preds = probs.argmax(dim=1)

    accuracy = skm.accuracy_score(labels, preds)
    precision = skm.precision_score(labels, preds, average="macro", zero_division=0)
    recall = skm.recall_score(labels, preds, average="macro", zero_division=0)
    f1 = skm.f1_score(labels, preds, average="macro", zero_division=0)

    loss = None
    if criterion is not None:
        loss = criterion(probs.log(), labels.to(probs.device)).item()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "loss": float(loss) if loss is not None else None,
    }


# -----------------------------------------------------------------------------
# Figure generation helpers
# -----------------------------------------------------------------------------

def _annotate_line(ax, x, y):
    for xi, yi in zip(x, y):
        ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points", xytext=(0, 5), ha="center", fontsize="x-small")


def _annotate_heatmap(ax, cm):
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black", fontsize="xx-small")


def generate_figures(history: Dict, figs_dir: Path) -> List[Path]:
    """Creates PDF figures for loss/accuracy curves + confusion matrix.

    Parameters
    ----------
    history
        Dict returned by :pyfunc:`src.train.train_model`.
    figs_dir
        Directory where figures are written to disk.

    Returns
    -------
    List[Path]
        Absolute paths to generated figure PDFs.
    """
    figs_dir.mkdir(exist_ok=True, parents=True)
    figure_paths: List[Path] = []

    # 1. Training & validation loss curve
    fig, ax = plt.subplots(figsize=(6, 4))
    epochs = history["epoch"]
    ax.plot(epochs, history["train_loss"], label="train")
    ax.plot(epochs, history["val_loss"], label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training vs. Validation Loss")
    ax.legend()
    _annotate_line(ax, epochs, history["train_loss"])
    _annotate_line(ax, epochs, history["val_loss"])
    loss_path = figs_dir / "training_loss.pdf"
    fig.tight_layout()
    fig.savefig(loss_path, bbox_inches="tight")
    plt.close(fig)
    figure_paths.append(loss_path)

    # 2. Training & validation accuracy curve
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, history["train_acc"], label="train")
    ax.plot(epochs, history["val_acc"], label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training vs. Validation Accuracy")
    ax.legend()
    _annotate_line(ax, epochs, history["train_acc"])
    _annotate_line(ax, epochs, history["val_acc"])
    acc_path = figs_dir / "accuracy.pdf"
    fig.tight_layout()
    fig.savefig(acc_path, bbox_inches="tight")
    plt.close(fig)
    figure_paths.append(acc_path)

    # 3. Confusion matrix if available
    if "confusion_matrix" in history:
        cm = np.asarray(history["confusion_matrix"], dtype=int)
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(cm, annot=False, fmt="d", ax=ax, cmap="Blues", cbar=True, square=True)
        _annotate_heatmap(ax, cm)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        cm_path = figs_dir / "confusion_matrix.pdf"
        fig.tight_layout()
        fig.savefig(cm_path, bbox_inches="tight")
        plt.close(fig)
        figure_paths.append(cm_path)

    return figure_paths
