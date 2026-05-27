from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a configuration file is missing required data or has invalid values."""


def _read_yaml(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise ConfigError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ConfigError(f"Configuration must be a mapping: {config_path}")
    return data


def _require_keys(config: dict[str, Any], required_keys: list[str], config_name: str) -> None:
    missing = [key for key in required_keys if key not in config]
    if missing:
        missing_csv = ", ".join(missing)
        raise ConfigError(f"Missing keys in {config_name}: {missing_csv}")


def load_env_config(config_path: str | Path, size: str | None = None) -> dict[str, Any]:
    path = Path(config_path)
    config = _read_yaml(path)

    # Merge named size variant (grid/start/goal) over the base config when requested.
    if size is not None:
        sizes = config.get("sizes", {})
        if size not in sizes:
            raise ConfigError(f"Unknown environment size '{size}'. Available: {list(sizes)}")
        config = {**config, **sizes[size]}

    _require_keys(config, ["grid", "flow", "start", "goal", "docking"], "env config")

    grid = config["grid"]
    if not isinstance(grid, dict) or "nx" not in grid or "ny" not in grid:
        raise ConfigError("'grid' must contain 'nx' and 'ny'.")

    if grid["nx"] <= 0 or grid["ny"] <= 0:
        raise ConfigError("Grid dimensions must be > 0.")

    return config


def load_algorithm_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    config = _read_yaml(path)
    _require_keys(
        config,
        [
            "common",
            "dijkstra",
            "a_star",
            "weighted_a_star",
            "dynamic_programming",
            "q_learning",
        ],
        "algorithm config",
    )

    weight = config["weighted_a_star"].get("weight")
    if weight is None or weight < 1:
        raise ConfigError("'weighted_a_star.weight' must be >= 1.")

    return config


def load_experiment_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    config = _read_yaml(path)
    _require_keys(config, ["seeds", "environments", "algorithms", "metrics"], "experiment config")

    if not isinstance(config["seeds"], list) or not config["seeds"]:
        raise ConfigError("'seeds' must be a non-empty list.")

    if not isinstance(config["algorithms"], list) or not config["algorithms"]:
        raise ConfigError("'algorithms' must be a non-empty list.")

    return config


def load_all_configs(config_dir: str | Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    base = Path(config_dir)
    env_cfg = load_env_config(base / "env.yaml")
    algo_cfg = load_algorithm_config(base / "algorithm.yaml")
    exp_cfg = load_experiment_config(base / "experiment.yaml")
    return env_cfg, algo_cfg, exp_cfg
