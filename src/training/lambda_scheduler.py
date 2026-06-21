from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LambdaSchedulerConfig:
    mode: str
    total_epochs: int
    warmup_epochs: int
    lambda_max: float
    lambda_min: float
    constant_lambda: float


def compute_lambda(config: LambdaSchedulerConfig, epoch: int) -> float:
    if config.total_epochs <= 0:
        raise ValueError("total_epochs must be positive")
    if epoch < 0:
        raise ValueError("epoch must be non-negative")

    mode = config.mode.lower()
    if mode == "constant":
        return float(config.constant_lambda)

    warmup_epochs = max(int(config.warmup_epochs), 0)
    lambda_max = float(config.lambda_max)
    lambda_min = float(config.lambda_min)
    total_epochs = int(config.total_epochs)

    if mode == "linear_warmup":
        if warmup_epochs <= 0:
            return lambda_max
        progress = min(epoch + 1, warmup_epochs) / warmup_epochs
        return lambda_max * progress

    if mode == "cosine_decay":
        if total_epochs == 1:
            return lambda_min
        progress = min(max(epoch, 0), total_epochs - 1) / (total_epochs - 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return lambda_min + (lambda_max - lambda_min) * cosine

    if mode == "warmup_cosine_decay":
        if warmup_epochs > 0 and epoch < warmup_epochs:
            progress = (epoch + 1) / warmup_epochs
            return lambda_max * progress
        decay_steps = max(total_epochs - warmup_epochs, 1)
        decay_epoch = min(max(epoch - warmup_epochs, 0), decay_steps - 1)
        progress = 1.0 if decay_steps == 1 else decay_epoch / (decay_steps - 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return lambda_min + (lambda_max - lambda_min) * cosine

    raise ValueError(f"Unsupported lambda schedule mode: {config.mode}")


class LambdaScheduler:
    def __init__(self, config: LambdaSchedulerConfig) -> None:
        self.config = config
        self.current_epoch = 0
        self.current_value = compute_lambda(config, epoch=0)

    def value_for_epoch(self, epoch: int) -> float:
        return compute_lambda(self.config, epoch=epoch)

    def step(self) -> float:
        self.current_epoch += 1
        next_epoch = min(self.current_epoch, max(self.config.total_epochs - 1, 0))
        self.current_value = compute_lambda(self.config, epoch=next_epoch)
        return self.current_value


def save_lambda_curve(curve: list[dict[str, float]], output_csv: str | Path, output_png: str | Path) -> None:
    csv_path = Path(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "lambda_sal"])
        writer.writeheader()
        writer.writerows(curve)

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [int(item["epoch"]) for item in curve]
    lambdas = [float(item["lambda_sal"]) for item in curve]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, lambdas, linewidth=2.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("lambda_sal")
    ax.set_title("Lambda Schedule")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png_path = Path(output_png)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
