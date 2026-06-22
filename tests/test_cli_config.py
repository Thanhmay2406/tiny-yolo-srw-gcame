from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.cli_config import parse_args_with_optional_config
from src.utils.io import save_yaml


def test_parse_args_with_optional_config_loads_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    save_yaml(
        config_path,
        {
            "epochs": 5,
            "run_name": "from_config",
        },
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--run-name", dest="run_name", type=str, required=True)

    args = parse_args_with_optional_config(parser, argv=["--config", str(config_path)])
    assert args.epochs == 5
    assert args.run_name == "from_config"


def test_parse_args_with_optional_config_allows_cli_override(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    save_yaml(
        config_path,
        {
            "epochs": 5,
            "run_name": "from_config",
        },
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--run-name", dest="run_name", type=str, required=True)

    args = parse_args_with_optional_config(
        parser,
        argv=["--config", str(config_path), "--epochs", "7", "--run-name", "from_cli"],
    )
    assert args.epochs == 7
    assert args.run_name == "from_cli"
