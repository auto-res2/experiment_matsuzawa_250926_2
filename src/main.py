"""src/main.py
Main entry point.  Provides CLI with --smoke-test / --full-experiment.
Loads configuration, launches training & evaluation, saves JSON results and
prints experiment description + numerical results to stdout.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml

# Local imports (module style: python -m src.main)
from . import preprocess as prep
from . import train as trainer
from .evaluate import evaluate, plot_training_curves

# -----------------------------------------------------------------------------
#                               UTILITIES
# -----------------------------------------------------------------------------

def _set_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
#                              CLI & DISPATCH
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Prototype Margin Regularised ProtoNet Experiments")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke-test", action="store_true", help="Run a quick synthetic test to validate pipeline")
    group.add_argument("--full-experiment", action="store_true", help="Run full experiment as defined in YAML config")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration. Defaults depend on the chosen mode.",
    )
    return parser.parse_args()


def load_config(args) -> Dict:
    if args.config is not None:
        cfg_path = Path(args.config)
    else:
        cfg_path = Path("config/smoke_test.yaml" if args.smoke_test else "config/full_experiment.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


# -----------------------------------------------------------------------------
#                                MAIN RUNNER
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg = load_config(args)

    _set_seed(cfg["training"].get("seed", 0))
    device = torch.device("cuda" if (torch.cuda.is_available() and cfg["training"].get("device", "cuda") == "cuda") else "cpu")

    # -------------------- data --------------------
    train_loader, val_loader, test_loader = prep.get_dataloaders(cfg)

    # -------------------- model -------------------
    model = trainer.get_backbone(cfg["model"]["name"], **{k: v for k, v in cfg["model"].items() if k != "name"})
    model.to(device)

    # -------------------- training ---------------
    start = time.time()
    best_model, history = trainer.train(model, train_loader, val_loader, cfg, device)
    training_time = time.time() - start

    # save training curves
    results_dir = Path(cfg["evaluation"].get("results_dir", "results"))
    figures_dir = results_dir / "figures"
    plot_training_curves(history, figures_dir)

    # -------------------- evaluation -------------
    metrics = evaluate(best_model, test_loader, cfg, device)
    metrics["training_time_sec"] = training_time

    # -------------------- save & print -----------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = cfg["dataset"]["name"] + "_" + cfg["model"]["name"]
    out_dir = results_dir / f"{exp_name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # stdout requirements
    print("=" * 80)
    print("Experiment description:")
    print(json.dumps(cfg, indent=4))
    print("\nNumerical results:")
    print(json.dumps(metrics, indent=4))
    print("\nFigures saved:")
    for pdf in figures_dir.glob("*.pdf"):
        print(" -", pdf.name)
    print("=" * 80)


if __name__ == "__main__":
    main()
