import argparse
import platform
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.device import get_device_summary
from src.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the local or Kaggle runtime for Phase 0.")
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments",
        help="Directory that should be writable for experiment outputs.",
    )
    return parser.parse_args()


def check_writable(path: Path) -> bool:
    try:
        ensure_dir(path)
        probe = path / ".write_test"
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def main() -> None:
    args = parse_args()
    device_summary = get_device_summary()
    experiments_dir = args.experiments_dir.resolve()

    print(f"Python version: {platform.python_version()}")
    print(f"torch version: {device_summary['torch_version']}")
    print(f"CUDA available: {device_summary['cuda_available']}")
    print(f"GPU name: {device_summary['gpu_name']}")
    print(f"Current working directory: {Path.cwd()}")
    print(f"experiments dir: {experiments_dir}")
    print(f"experiments writable: {check_writable(experiments_dir)}")


if __name__ == "__main__":
    main()
