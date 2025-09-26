"""src/main.py
Command-line front-end for the common experiment framework.
Supports --smoke-test and --full-experiment as required.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Dict

import torch
from torch import nn

from .preprocess import get_dataloaders
from .train import train_model

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def _load_yaml(fp: Path) -> Dict:
    import yaml  # local import keeps base import list minimal

    with fp.open("r") as f:
        return yaml.safe_load(f)


def _build_model(config: Dict, num_classes: int) -> nn.Module:
    name: str = config["model"]["name"].lower()

    if name == "simplenet":
        import torch.nn as nn

        class SimpleNet(nn.Module):
            def __init__(self, classes: int):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(64, classes),
                )

            def forward(self, x):
                return self.net(x)

        return SimpleNet(num_classes)

    elif name == "resnet18":
        import torchvision.models as models
        import torch.nn as nn
        model = models.resnet18(pretrained=False)
        # Replace the final fully connected layer for CIFAR-10 classes
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    # ------------------------------------------------------------------
    # PLACEHOLDER: Will be replaced by task-specific model constructor.
    # ------------------------------------------------------------------
    raise NotImplementedError(f"Model '{name}' not yet implemented in framework.")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Common Core Experiment Runner")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke-test", action="store_true", help="Run quick synthetic test")
    group.add_argument("--full-experiment", action="store_true", help="Run full experiment from config")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML config (overrides default for each mode)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # ------------------------------------------------------------------
    # Resolve configuration file
    # ------------------------------------------------------------------
    root_dir = Path(__file__).resolve().parent.parent
    if args.config:
        cfg_path = Path(args.config)
    else:
        cfg_path = (
            root_dir / "config" / ("smoke_test.yaml" if args.smoke_test else "full_experiment.yaml")
        )

    if not cfg_path.exists():
        print(f"[ERROR] Configuration file '{cfg_path}' not found.")
        sys.exit(1)

    config: Dict = _load_yaml(cfg_path)

    torch.manual_seed(int(config["training"].get("seed", 42)))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Prepare directories
    # ------------------------------------------------------------------
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = (
        Path(config["training"].get("save_dir", "experiments"))
        / f"{config['dataset']['name']}"  # type: ignore[index]
        / f"{config['model']['name']}"  # type: ignore[index]
        / ts
    ).resolve()
    exp_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Data & Model
    # ------------------------------------------------------------------
    train_loader, val_loader, test_loader, num_classes = get_dataloaders(config)
    model = _build_model(config, num_classes)

    # ------------------------------------------------------------------
    # Run training
    # ------------------------------------------------------------------
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        experiment_dir=exp_dir,
    )

    # ------------------------------------------------------------------
    # Evaluate on test set using the *best* checkpoint
    # ------------------------------------------------------------------
    best_ckpt = Path(history["best_ckpt_path"])
    if best_ckpt.exists():
        from src.train import _load_checkpoint  # avoid circular import at top

        model, _ = _load_checkpoint(model, best_ckpt, map_location=device)
    else:
        print("[WARN] Best checkpoint not found. Using current model weights.")

    from src.evaluate import evaluate_model

    criterion = torch.nn.CrossEntropyLoss()
    test_metrics = evaluate_model(model, test_loader, device, criterion)

    # confusion matrix to history
    with torch.no_grad():
        import numpy as np
        from sklearn.metrics import confusion_matrix

        y_probs, y_true = [], []
        for inputs, targets in test_loader:
            y_true.append(targets)
            y_probs.append(model(inputs.to(device)).argmax(dim=1).cpu())
        y_true = torch.cat(y_true)
        y_pred = torch.cat(y_probs)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
        history["confusion_matrix"] = cm.tolist()

    history["test_metrics"] = test_metrics

    # Update results.json with test metrics
    results_path = exp_dir / "results.json"
    with results_path.open("w") as fp:
        json.dump(history, fp, indent=2)

    # ------------------------------------------------------------------
    # Standard-output requirements
    # ------------------------------------------------------------------
    print("\n================= EXPERIMENT DESCRIPTION =================")
    print(config.get("description", "<No description in config>"))
    print("\n================= EXPERIMENTAL RESULTS ===================")
    print(json.dumps(history, indent=2))
    print("\nFigures generated (PDF):")
    for fig_path in history["figures"]:
        print(" - ", fig_path)


if __name__ == "__main__":
    main()
