from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .learning_rules import (
    AngleSummary,
    accuracy_from_positive_output,
    apply_deltas,
    biological_bias_deltas,
    biological_weight_deltas,
    bp_bias_deltas,
    bp_weight_deltas,
    weight_delta_angles,
)
from .model import PositiveMLP


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_accuracy: float
    mean_angle_degrees: Optional[float]
    layer_angle_degrees: List[Optional[float]]


def mnist_loader(
    data_dir: Path,
    batch_size: int,
    train: bool,
    limit_samples: Optional[int],
) -> DataLoader:
    dataset = datasets.MNIST(
        root=str(data_dir),
        train=train,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda tensor: tensor.view(-1)),
            ]
        ),
    )
    if limit_samples is not None:
        dataset = Subset(dataset, range(min(limit_samples, len(dataset))))
    return DataLoader(dataset, batch_size=batch_size, shuffle=train)


def average_angles(summaries: List[AngleSummary]) -> AngleSummary:
    if not summaries:
        return AngleSummary(layer_angles=[], mean_angle=None)

    layer_count = len(summaries[0].layer_angles)
    layer_means: List[Optional[float]] = []
    for layer_index in range(layer_count):
        values = [
            summary.layer_angles[layer_index]
            for summary in summaries
            if summary.layer_angles[layer_index] is not None
        ]
        layer_means.append(float(sum(values) / len(values)) if values else None)

    valid_layer_means = [angle for angle in layer_means if angle is not None]
    mean_angle = (
        float(sum(valid_layer_means) / len(valid_layer_means))
        if valid_layer_means
        else None
    )
    return AngleSummary(layer_angles=layer_means, mean_angle=mean_angle)


def run_epoch(
    model: PositiveMLP,
    loader: DataLoader,
    device: torch.device,
    learning_rate: float,
    train_rule: str,
    scale_mode: str,
    update_function: str,
    update_bias: bool,
) -> EpochMetrics:
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    total_examples = 0
    angle_summaries: List[AngleSummary] = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        batch_size = targets.shape[0]

        cache = model.forward_with_cache(inputs)
        loss = model.loss_from_output(cache.activations[-1], targets)

        bp_deltas = bp_weight_deltas(
            model=model,
            loss=loss,
            learning_rate=learning_rate,
            retain_graph=True,
        )
        bio_deltas = biological_weight_deltas(
            model=model,
            cache=cache,
            targets=targets,
            learning_rate=learning_rate,
            scale_mode=scale_mode,
            update_function=update_function,
        )
        angle_summaries.append(weight_delta_angles(bp_deltas, bio_deltas))

        if train_rule == "bp":
            bias_deltas = (
                bp_bias_deltas(model, loss, learning_rate, retain_graph=False)
                if update_bias
                else None
            )
            apply_deltas(model, bp_deltas, bias_deltas)
        elif train_rule == "biological":
            bias_deltas = (
                biological_bias_deltas(
                    model=model,
                    cache=cache,
                    targets=targets,
                    learning_rate=learning_rate,
                    scale_mode=scale_mode,
                    update_function=update_function,
                )
                if update_bias
                else None
            )
            apply_deltas(model, bio_deltas, bias_deltas)
        elif train_rule != "none":
            raise ValueError(f"unknown train rule: {train_rule}")

        total_loss += float(loss.detach().item()) * batch_size
        total_accuracy += (
            accuracy_from_positive_output(cache.activations[-1], targets) * batch_size
        )
        total_examples += batch_size

    angle_summary = average_angles(angle_summaries)
    return EpochMetrics(
        epoch=0,
        train_loss=total_loss / total_examples,
        train_accuracy=total_accuracy / total_examples,
        mean_angle_degrees=angle_summary.mean_angle,
        layer_angle_degrees=angle_summary.layer_angles,
    )


def save_metrics_csv(metrics: List[EpochMetrics], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    max_layers = max((len(metric.layer_angle_degrees) for metric in metrics), default=0)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "epoch",
            "train_loss",
            "train_accuracy",
            "mean_angle_degrees",
        ] + [f"layer_{index + 1}_angle_degrees" for index in range(max_layers)]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for metric in metrics:
            row = {
                "epoch": metric.epoch,
                "train_loss": metric.train_loss,
                "train_accuracy": metric.train_accuracy,
                "mean_angle_degrees": metric.mean_angle_degrees,
            }
            for index, angle in enumerate(metric.layer_angle_degrees):
                row[f"layer_{index + 1}_angle_degrees"] = angle
            writer.writerow(row)


def save_angle_plot(metrics: List[EpochMetrics], path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [metric.epoch for metric in metrics]
    mean_angles = [metric.mean_angle_degrees for metric in metrics]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, mean_angles, marker="o", label="Mean layer angle")
    if metrics and metrics[0].layer_angle_degrees:
        for layer_index in range(len(metrics[0].layer_angle_degrees)):
            values = [
                metric.layer_angle_degrees[layer_index]
                for metric in metrics
            ]
            plt.plot(
                epochs,
                values,
                linestyle="--",
                alpha=0.55,
                label=f"Layer {layer_index + 1}",
            )
    plt.xlabel("Epoch")
    plt.ylabel("Angle between update vectors (degrees)")
    plt.title("BP vs biological-rule weight-change angle on MNIST")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def run_experiment(args: argparse.Namespace) -> List[EpochMetrics]:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = PositiveMLP(
        input_dim=28 * 28,
        hidden_dim=args.hidden_dim,
        output_dim=10,
        hidden_layers=args.hidden_layers,
    ).to(device)

    loader = mnist_loader(
        data_dir=Path(args.data_dir),
        batch_size=args.batch_size,
        train=True,
        limit_samples=args.limit_samples,
    )

    all_metrics: List[EpochMetrics] = []
    for epoch in range(1, args.epochs + 1):
        metric = run_epoch(
            model=model,
            loader=loader,
            device=device,
            learning_rate=args.learning_rate,
            train_rule=args.train_rule,
            scale_mode=args.scale_mode,
            update_function=args.update_function,
            update_bias=args.update_bias,
        )
        metric.epoch = epoch
        all_metrics.append(metric)
        mean_angle = (
            f"{metric.mean_angle_degrees:.3f}"
            if metric.mean_angle_degrees is not None
            else "nan"
        )
        print(
            f"epoch={epoch:03d} "
            f"loss={metric.train_loss:.4f} "
            f"acc={metric.train_accuracy:.4f} "
            f"mean_angle={mean_angle}"
        )

    save_metrics_csv(all_metrics, Path(args.output_csv))
    save_angle_plot(all_metrics, Path(args.output_plot))
    return all_metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare standard BP and the proposed positive-activation biological "
            "learning rule on MNIST."
        )
    )
    parser.add_argument("--data-dir", default="data", help="MNIST download/cache path")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cpu, cuda, or mps",
    )
    parser.add_argument(
        "--train-rule",
        choices=["biological", "bp", "none"],
        default="biological",
        help="Which update rule is applied after both candidate updates are measured.",
    )
    parser.add_argument("--scale-mode", choices=["tanh", "clamp"], default="tanh")
    parser.add_argument(
        "--update-function",
        choices=["epsilon_times_activation", "epsilon", "activity_gated"],
        default="epsilon_times_activation",
    )
    parser.add_argument(
        "--update-bias",
        action="store_true",
        help="Also update biases. Weight-angle comparison always uses weights only.",
    )
    parser.add_argument(
        "--limit-samples",
        type=int,
        default=None,
        help="Use a prefix of MNIST for quick smoke runs.",
    )
    parser.add_argument("--output-csv", default="outputs/angle_metrics.csv")
    parser.add_argument("--output-plot", default="outputs/angle_trend.png")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
