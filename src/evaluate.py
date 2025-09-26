"""src/evaluate.py
Evaluation utilities shared across experiments – metrics, JSON result
merging and *publication-quality* figure generation.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

sns.set_theme(style="whitegrid")
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})  # embed fonts

__all__ = [
    "AccuracyMeter",
    "ConfusionMatrixMeter",
    "make_figures",
]


class AccuracyMeter:
    """Tracks running accuracy and stores full history (for curves)."""

    def __init__(self):
        self.correct = 0
        self.count = 0
        self._history: List[float] = []  # accuracy after each update

    # ------------------------------------------------------------------ #
    def update(self, preds, target):
        preds = np.asarray(preds)
        target = np.asarray(target)
        self.correct += (preds == target).sum()
        self.count += preds.size
        self._history.append(self.accuracy())

    def accuracy(self) -> float:  # returns [0,1]
        return self.correct / max(1, self.count)

    # expose accuracy history
    @property
    def accuracy_history(self) -> List[float]:
        return self._history


class ConfusionMatrixMeter:
    """Accumulate confusion matrix incrementally to avoid keeping all preds."""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, preds, target):
        cm = confusion_matrix(target, preds, labels=list(range(self.num_classes)))
        self.matrix += cm

    def compute(self):
        return self.matrix


# --------------------------------------------------------------------------- #
#                             Figure utilities                                #
# --------------------------------------------------------------------------- #

def _annotate_line(ax, x, y):
    for xv, yv in zip(x, y):
        ax.annotate(f"{yv*100:.1f}", xy=(xv, yv), xytext=(0, 4), textcoords="offset points", fontsize=6)


def make_figures(
    losses: List[float],
    accuracies: List[float],
    conf_mat: np.ndarray,
    out_dir: Path,
) -> List[Path]:
    out_dir.mkdir(exist_ok=True, parents=True)
    paths: List[Path] = []

    # 1. Training loss curve
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = list(range(1, len(losses) + 1))
    ax.plot(xs, losses, label="Entropy loss", color="tab:blue")
    ax.set_xlabel("Batch index")
    ax.set_ylabel("Loss (nats)")
    ax.set_title("Training loss during streaming adaptation")
    for xv, yv in zip(xs, losses):
        ax.annotate(f"{yv:.2f}", (xv, yv), textcoords="offset points", xytext=(0, 4), fontsize=6)
    ax.legend()
    fname = out_dir / "training_loss.pdf"
    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight")
    paths.append(fname)
    plt.close(fig)

    # 2. Accuracy curve
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = list(range(1, len(accuracies) + 1))
    ax.plot(xs, accuracies, label="Top-1 accuracy", color="tab:green")
    ax.set_xlabel("Batch index")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    for xv, yv in zip(xs, accuracies):
        _annotate_line(ax, [xv], [yv])
    ax.set_title("Accuracy curve")
    ax.legend()
    fname = out_dir / "accuracy.pdf"
    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight")
    paths.append(fname)
    plt.close(fig)

    # 3. Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    norm = conf_mat / conf_mat.sum(axis=1, keepdims=True).clip(min=1)
    sns.heatmap(norm, annot=False, fmt=".2f", cmap="Blues", ax=ax, cbar_kws={"label": "Recall"})
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Normalized confusion matrix")
    fname = out_dir / "confusion_matrix.pdf"
    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight")
    paths.append(fname)
    plt.close(fig)

    return paths
