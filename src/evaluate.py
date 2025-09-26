"""src/evaluate.py
Universal evaluation framework shared across all experimental variations.
Produces numerical metrics, confidence intervals and publication-quality
figures (PDF) with exhaustive annotations.
"""
from __future__ import annotations

import math
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm

# -----------------------  METRIC COMPUTATIONS  ------------------------------

def _accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    return (pred == target).float().mean().item() * 100.0


def _confidence_interval(data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    arr = np.array(data)
    mean = arr.mean()
    sem = stats.sem(arr)
    df = len(arr) - 1
    t_val = stats.t.ppf((1 + confidence) / 2.0, df)
    return mean, t_val * sem


def _expected_calibration_error(
    probs: np.ndarray, targets: np.ndarray, n_bins: int = 15
) -> float:
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences = probs.max(1)
    predictions = probs.argmax(1)
    accuracies = (predictions == targets).astype(float)

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece * 100.0  # percentage


# -----------------------  FIGURE HELPERS  -----------------------------------

def _annotate_bars(ax):
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.2f}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_training_curves(history: Dict[str, List[float]], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    # Loss curve
    plt.figure()
    plt.plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.xticks(epochs)
    for x, y in zip(epochs, history["train_loss"]):
        plt.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 5), ha="center")
    plt.legend()
    fname = out_dir / "training_loss.pdf"
    plt.savefig(fname, bbox_inches="tight")
    plt.close()

    # Accuracy curve (if available)
    if "val_acc" in history and len(history["val_acc"]):
        plt.figure()
        plt.plot(epochs, history["val_acc"], marker="o", c="green", label="Val Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title("Validation Accuracy Curve")
        plt.xticks(epochs)
        for x, y in zip(epochs, history["val_acc"]):
            plt.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 5), ha="center")
        plt.legend()
        fname = out_dir / "accuracy.pdf"
        plt.savefig(fname, bbox_inches="tight")
        plt.close()


def plot_confusion(cm: np.ndarray, class_names: List[str], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    fname = out_dir / "confusion_matrix.pdf"
    plt.savefig(fname, bbox_inches="tight")
    plt.close()


# -----------------------  MAIN EVALUATION  ----------------------------------

def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    cfg: Dict,
    device: torch.device,
    *,
    disable_figures: bool = False,
) -> Dict[str, float]:
    model.eval()
    way = cfg["algo"]["way"]
    temperature = cfg["algo"].get("temperature", 10.0)

    episode_accuracies: List[float] = []
    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Evaluation"):
            imgs = imgs.to(device)
            labels = labels.to(device)
            embeddings = model(imgs)

            n_support = embeddings.size(0) // 2
            emb_sup, emb_q = embeddings[:n_support], embeddings[n_support:]
            lbl_sup, lbl_q = labels[:n_support], labels[n_support:]

            protos = []
            for c in range(way):
                protos.append(emb_sup[lbl_sup == c].mean(0))
            protos = F.normalize(torch.stack(protos), dim=1)
            logits = F.normalize(emb_q, dim=1) @ protos.t()
            logits = logits * temperature
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(1)
            acc = _accuracy(preds, lbl_q)
            episode_accuracies.append(acc)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(lbl_q.cpu().numpy())

    mean_acc, half_ci = _confidence_interval(episode_accuracies, cfg["evaluation"]["confidence_level"])
    all_probs_np = np.concatenate(all_probs, axis=0)
    all_labels_np = np.concatenate(all_labels, axis=0)
    ece = _expected_calibration_error(all_probs_np, all_labels_np)
    cm = confusion_matrix(all_labels_np, all_probs_np.argmax(1))

    metrics = {
        "mean_accuracy": mean_acc,
        "ci95": half_ci,
        "ece": ece,
    }

    if not disable_figures:
        out_dir = Path(cfg["evaluation"].get("results_dir", "results")) / "figures"
        plot_confusion(cm, [f"C{i}" for i in range(way)], out_dir)
    return metrics
