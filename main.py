from __future__ import annotations

import argparse

from core.config_loader import load_all_configs
from ui.visualization import launch_ui


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="River Crossing CLI")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("check", help="Validate and print key config values")

    sub.add_parser("ui", help="Launch the PyQt6 desktop UI")

    return parser


def _run_check_command() -> int:
    env_cfg, _, exp_cfg = load_all_configs("configs")
    print("Configs loaded successfully")
    print(f"Grid: {env_cfg['grid']['nx']}x{env_cfg['grid']['ny']}")
    print(f"Algorithms: {len(exp_cfg['algorithms'])}")
    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    # Keep previous default behavior when no command is provided.
    if args.command is None or args.command == "check":
        return _run_check_command()

    if args.command == "ui":
        return launch_ui()

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
