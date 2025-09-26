"""src/train.py
Core training loop, model checkpointing, and history tracking.
This file contains the full implementation of the framework-level
training logic that is shared by all experimental variations.
Only dataset/model–specific pieces live behind clearly marked
PLACEHOLDER comments so they can be swapped in the next phase.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from .evaluate import evaluate_model, generate_figures

__all__ = [
    "train_model",
]


def _save_checkpoint(
    state: Dict,
    ckpt_path: Path,
) -> None:
    """Serialises *state* dict to *ckpt_path* using :pyfunc:`torch.save`."""
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, ckpt_path)


def _load_checkpoint(
    model: nn.Module,
    ckpt_path: Path,
    map_location: str | torch.device = "cpu",
) -> Tuple[nn.Module, Dict]:
    """Loads checkpoint into *model* and returns (model, state_dict)."""
    state = torch.load(ckpt_path, map_location=map_location)
    model.load_state_dict(state["model_state"])
    return model, state


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict,
    device: torch.device,
    experiment_dir: Path,
) -> Dict:
    """High-level training entry point.

    Parameters
    ----------
    model
        Instantiated PyTorch model.
    train_loader / val_loader
        DataLoaders obtained from :pymod:`src.preprocess`.
    config
        Parsed YAML configuration as a Python dict.
    device
        CUDA / CPU device to train on.
    experiment_dir
        Root directory where checkpoints, logs, and figures are written.

    Returns
    -------
    history
        Dict containing loss / accuracy curves, best metrics, runtime, etc.
    """
    epochs: int = int(config["training"]["epochs"])
    lr: float = float(config["model"].get("lr", 1e-3))
    wd: float = float(config["model"].get("weight_decay", 0.0))
    patience: int = int(config["training"].get("early_stopping_patience", 10))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience // 2)

    history: Dict[str, List] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "epoch": [],
    }

    best_val_acc: float = -float("inf")
    best_epoch: int = -1
    start = time.time()

    ckpt_dir = experiment_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / "best_model.pt"
    last_ckpt_path = ckpt_dir / "last_model.pt"

    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        total_samples = 0

        for batch in train_loader:
            inputs, targets = batch
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(dim=1)
            correct = (preds == targets).sum().item()

            running_loss += loss.item() * inputs.size(0)
            running_correct += correct
            total_samples += inputs.size(0)

        train_loss_epoch = running_loss / total_samples
        train_acc_epoch = running_correct / total_samples

        val_metrics = evaluate_model(model, val_loader, device, criterion)
        val_loss_epoch = val_metrics["loss"]
        val_acc_epoch = val_metrics["accuracy"]

        scheduler.step(val_loss_epoch)

        # Logging
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss_epoch)
        history["val_loss"].append(val_loss_epoch)
        history["train_acc"].append(train_acc_epoch)
        history["val_acc"].append(val_acc_epoch)

        # Checkpointing
        if val_acc_epoch > best_val_acc:
            best_val_acc = val_acc_epoch
            best_epoch = epoch
            _save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_val_acc": best_val_acc,
                },
                best_ckpt_path,
            )

        # Always save last epoch as well
        _save_checkpoint(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
            },
            last_ckpt_path,
        )

        # Console progress
        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"Train Loss: {train_loss_epoch:.4f}  Acc: {train_acc_epoch:.4f} | "
            f"Val Loss: {val_loss_epoch:.4f}  Acc: {val_acc_epoch:.4f}"
        )

    runtime = time.time() - start

    # Attach summary fields
    history["best_val_acc"] = best_val_acc
    history["best_epoch"] = best_epoch
    history["runtime_sec"] = runtime
    history["best_ckpt_path"] = str(best_ckpt_path)

    # Plot historical curves & save
    figs_dir = experiment_dir / "figures"
    figs_dir.mkdir(exist_ok=True, parents=True)
    figure_paths = generate_figures(history, figs_dir)

    history["figures"] = [str(p) for p in figure_paths]

    # Dump history to JSON
    results_path = experiment_dir / "results.json"
    with results_path.open("w") as fp:
        json.dump(history, fp, indent=2)

    print("\n===== Training Complete =====")
    print(f"Best Validation Accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
    print(f"Total Runtime: {runtime/60:.2f} min")

    return history
