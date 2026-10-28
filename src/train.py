"""
Training entry point for the EuroSAT multispectral classification project.

Run from the project root:
    python -u -m src.train

The script trains a SmallCNN baseline, selects the best checkpoint using
validation loss, applies early stopping, and saves the class mapping and metrics.
"""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW


# Make the project root available for imports when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import get_loaders
from src.model import SmallCNN


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(requested: str) -> torch.device:
    """Select CPU or GPU."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(requested)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but no CUDA-enabled GPU is available. "
            "Use --device cpu or --device auto."
        )

    return device


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    phase="Training",
):
    """Run one training or evaluation epoch."""
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_batches = len(loader)

    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for batch_index, (images, labels) in enumerate(loader, start=1):

            # Show progress so slow batches are easier to identify.
            if (
                batch_index == 1
                or batch_index % 10 == 0
                or batch_index == total_batches
            ):
                print(
                    f"{phase}: batch {batch_index}/{total_batches}",
                    flush=True,
                )

            images = images.to(device, non_blocking=True)
            labels = labels.to(
                device,
                dtype=torch.long,
                non_blocking=True,
            )

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_correct += (
                logits.argmax(dim=1) == labels
            ).sum().item()
            total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError(
            f"The {phase.lower()} data loader returned no samples. "
            "Check the dataset and data split."
        )

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


def save_json(path: Path, payload) -> None:
    """Save results as a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def parse_args():
    """Read command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a SmallCNN on EuroSAT multispectral imagery."
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "EuroSAT_MS",
        help="Dataset directory containing one subfolder per class.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "baseline",
    )

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-channels", type=int, default=13)

    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )

    parser.add_argument(
        "--splits-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "splits.csv",
        help="Path for the dataset split CSV.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")

    if args.patience < 1:
        raise ValueError("--patience must be at least 1.")

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    if args.num_channels < 1:
        raise ValueError("--num-channels must be at least 1.")

    set_seed(args.seed)
    device = get_device(args.device)

    data_dir = args.data_dir

    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir

    output_dir = args.output_dir

    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Project root : {PROJECT_ROOT}", flush=True)
    print(f"Dataset      : {data_dir}", flush=True)
    print(f"Output       : {output_dir}", flush=True)
    print(f"Device       : {device}", flush=True)
    print(f"Seed         : {args.seed}", flush=True)

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory was not found: {data_dir}\n"
            "Pass the correct folder using --data-dir."
        )

    print("\nLoading dataset and creating data loaders...", flush=True)

    train_loader, val_loader, test_loader, classes = get_loaders(
        data_dir=data_dir,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        num_channels=args.num_channels,
        splits_path=args.splits_path,
    )

    if not classes:
        raise RuntimeError(
            "No class names were returned by get_loaders()."
        )

    # Save the mapping between class indices and class names.
    save_json(
        output_dir / "classes.json",
        {
            str(index): class_name
            for index, class_name in enumerate(classes)
        },
    )

    print("Creating model and optimizer...", flush=True)

    model = SmallCNN(
        num_classes=len(classes),
        in_channels=args.num_channels,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    checkpoint_path = output_dir / "best_model.pth"
    history_path = output_dir / "metrics.csv"
    summary_path = output_dir / "metrics.json"

    history = []
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    start_time = time.time()

    print(f"Classes      : {len(classes)}", flush=True)
    print(f"Train batches: {len(train_loader)}", flush=True)
    print(f"Val batches  : {len(val_loader)}", flush=True)
    print(f"Test batches : {len(test_loader)}", flush=True)
    print("-" * 72, flush=True)

    print("Starting training loop...", flush=True)

    for epoch in range(1, args.epochs + 1):

        epoch_start = time.time()

        print(
            f"\nStarting Epoch {epoch}/{args.epochs}",
            flush=True,
        )

        # Training phase
        print("Training: processing batches...", flush=True)

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            phase="Training",
        )

        print(
            "Training finished. Starting validation...",
            flush=True,
        )

        # Validation phase
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            phase="Validation",
        )

        print("Validation finished.", flush=True)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "epoch_seconds": time.time() - epoch_start,
        }

        history.append(row)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss: {row['train_loss']:.4f} | "
            f"train acc: {row['train_accuracy']:.4f} | "
            f"val loss: {row['val_loss']:.4f} | "
            f"val acc: {row['val_accuracy']:.4f} | "
            f"{row['epoch_seconds']:.1f}s",
            flush=True,
        )

        # Save the model when validation loss improves.
        improved = (
            val_metrics["loss"]
            < best_val_loss - args.min_delta
        )

        if improved:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "val_accuracy": val_metrics["accuracy"],
                    "classes": classes,
                    "num_channels": args.num_channels,
                    "seed": args.seed,
                    "model_name": "SmallCNN",
                },
                checkpoint_path,
            )

            print("Best model checkpoint saved.", flush=True)

        else:
            epochs_without_improvement += 1

        # Save the training history after each epoch.
        with history_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(history[0].keys()),
            )
            writer.writeheader()
            writer.writerows(history)

        # Stop when validation loss no longer improves.
        if epochs_without_improvement >= args.patience:
            print(
                "Early stopping: validation loss did not improve "
                f"for {args.patience} consecutive epochs.",
                flush=True,
            )
            break

    if not checkpoint_path.exists():
        raise RuntimeError(
            "Training ended without saving a checkpoint."
        )

    # Evaluate the best checkpoint on the test set.
    print(
        "\nLoading best checkpoint for final test evaluation...",
        flush=True,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = run_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        optimizer=None,
        phase="Test",
    )

    total_seconds = time.time() - start_time

    summary = {
        "model": "SmallCNN",
        "dataset": str(data_dir),
        "num_channels": args.num_channels,
        "num_classes": len(classes),
        "classes": classes,
        "seed": args.seed,
        "device": str(device),
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "best_validation_accuracy": checkpoint["val_accuracy"],
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "epochs_completed": len(history),
        "training_seconds": round(total_seconds, 2),
        "hyperparameters": {
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "min_delta": args.min_delta,
        },
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "classes": str(output_dir / "classes.json"),
            "history": str(history_path),
        },
    }

    save_json(summary_path, summary)

    print("-" * 72, flush=True)
    print(f"Best epoch   : {best_epoch}", flush=True)
    print(f"Test loss    : {test_metrics['loss']:.4f}", flush=True)
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}", flush=True)
    print(f"Checkpoint   : {checkpoint_path}", flush=True)
    print(f"Class mapping: {output_dir / 'classes.json'}", flush=True)
    print(f"Metrics CSV  : {history_path}", flush=True)
    print(f"Metrics JSON : {summary_path}", flush=True)


if __name__ == "__main__":
    main()
