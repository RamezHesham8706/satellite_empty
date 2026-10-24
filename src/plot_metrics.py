"""
Plot training and validation metrics.

Run from the project root:
    python -u -m src.plot_metrics
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

METRICS_PATH = (
    PROJECT_ROOT / "outputs" / "baseline" / "metrics.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation"


def main():
    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            f"Training metrics file not found:\n{METRICS_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with METRICS_PATH.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise RuntimeError("The metrics CSV file is empty.")

    epochs = [int(row["epoch"]) for row in rows]

    train_loss = [
        float(row["train_loss"]) for row in rows
    ]
    val_loss = [
        float(row["val_loss"]) for row in rows
    ]

    train_accuracy = [
        float(row["train_accuracy"]) * 100 for row in rows
    ]
    val_accuracy = [
        float(row["val_accuracy"]) * 100 for row in rows
    ]

    # Plot 1: Loss
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(epochs, train_loss, marker="o", label="Training Loss")
    ax.plot(epochs, val_loss, marker="o", label="Validation Loss")

    ax.set_title("Training and Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_xticks(epochs)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "loss_curves.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Plot 2: Accuracy
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        epochs,
        train_accuracy,
        marker="o",
        label="Training Accuracy",
    )
    ax.plot(
        epochs,
        val_accuracy,
        marker="o",
        label="Validation Accuracy",
    )

    ax.set_title("Training and Validation Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(epochs)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "accuracy_curves.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    best_loss_row = min(rows, key=lambda row: float(row["val_loss"]))
    best_accuracy_row = max(
        rows, key=lambda row: float(row["val_accuracy"])
    )

    print("Training plots saved successfully.")
    print(f"Epochs completed: {len(epochs)}")
    print(
        "Best validation-loss epoch:",
        best_loss_row["epoch"],
    )
    print(
        "Best validation accuracy:",
        f"{float(best_accuracy_row['val_accuracy']) * 100:.2f}%",
        "at epoch",
        best_accuracy_row["epoch"],
    )
    print(f"Loss plot: {OUTPUT_DIR / 'loss_curves.png'}")
    print(f"Accuracy plot: {OUTPUT_DIR / 'accuracy_curves.png'}")


if __name__ == "__main__":
    main()
