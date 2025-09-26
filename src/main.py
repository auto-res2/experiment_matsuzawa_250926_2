"""src/main.py
Command-line interface.  Two entry modes:
    • --smoke-test       minimal run on synthetic data (fast)
    • --full-experiment  full run according to configuration

Example usage:
    uv run python -m src.main --smoke-test
"""
from __future__ import annotations

import argparse
import math
import pprint
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

# Ensure relative imports work when executed as module (-m src.main)
if __package__ is None or __package__ == "":
    import importlib.util
    import pathlib
    import sys as _sys

    file = pathlib.Path(__file__).resolve()
    parent, _ = file.parent, file.stem
    _sys.path.append(str(parent.parent))

from src.train import run_experiment  # noqa: E402  pylint: disable=wrong-import-position

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def load_config(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def describe_experiment(cfg: Dict[str, Any]):
    desc_lines = [
        f"Experiment: {cfg['experiment_name']}",
        f"Dataset: {cfg['dataset']['name']}",
        f"Model:   {cfg['model']['name']}",
        f"Method:  {cfg['adaptation']['method']}",
        "--- Hyper-parameters ---",
    ]
    for k, v in cfg["adaptation"].items():
        desc_lines.append(f"  {k}: {v}")
    return "\n".join(desc_lines)


def main():  # noqa: D401
    parser = argparse.ArgumentParser(description="Common-Core Experiment Runner")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke-test", action="store_true", help="Run a minimal end-to-end test.")
    g.add_argument("--full-experiment", action="store_true", help="Run the full experiment as defined in config/full_experiment.yaml")
    parser.add_argument("--config", type=str, default=None, help="Path to a custom YAML config (overrides preset)")
    args = parser.parse_args()

    if args.config:
        cfg_path = Path(args.config)
        assert cfg_path.exists(), f"Config file not found: {cfg_path}"
    else:
        cfg_path = CONFIG_DIR / ("smoke_test.yaml" if args.smoke_test else "full_experiment.yaml")

    cfg = load_config(cfg_path)

    # Print experiment description BEFORE results
    print("================ EXPERIMENT DESCRIPTION ================")
    print(describe_experiment(cfg))
    print("========================================================\n")

    results = run_experiment(cfg)

    # ----------------  Output results to stdout  ---------------- #
    print("================ EXPERIMENTAL RESULTS =================")
    pprint.pprint(results)
    print("=======================================================")


if __name__ == "__main__":
    main()
