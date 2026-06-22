from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.utils.io import load_yaml


def parse_args_with_optional_config(
    parser: argparse.ArgumentParser,
    argv: list[str] | None = None,
) -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path, default=None)
    bootstrap_args, remaining = bootstrap.parse_known_args(argv)

    config_path = bootstrap_args.config
    if config_path is not None:
        resolved = config_path.expanduser()
        if not resolved.is_absolute():
            resolved = (Path.cwd() / resolved).resolve()
        if not resolved.is_file():
            raise SystemExit(f"Config YAML not found: {resolved}")
        payload = load_yaml(resolved)
        if not isinstance(payload, dict):
            raise SystemExit(f"Config YAML must contain a mapping: {resolved}")
        parser.set_defaults(**payload)
        config_keys = {str(key) for key in payload}
        for action in parser._actions:
            if getattr(action, "required", False) and action.dest in config_keys:
                action.required = False

    namespace = parser.parse_args(remaining)
    setattr(namespace, "config", config_path)
    return namespace


def namespace_to_config_reference(args: Any) -> str | None:
    config_path = getattr(args, "config", None)
    if config_path is None:
        return None
    resolved = Path(config_path).expanduser()
    if not resolved.is_absolute():
        resolved = (Path.cwd() / resolved).resolve()
    return str(resolved)
