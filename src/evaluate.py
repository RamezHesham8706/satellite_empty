"""
Evaluate a trained EuroSAT multispectral classification model.

Run from the project root:
    python -u -m src.evaluate
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import get_loaders
from src.model import SmallCNN



# Configuration

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "EuroSAT_MS"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation"
CHECKPOINT_PATH = (
    PROJECT_ROOT / "outputs" / "baseline" / "best_model.pth"
)
SPLITS_PATH = PROJECT_ROOT / "data" / "processed" / "splits.csv"

BATCH_SIZE = 32
SEED = 42
NUM_CHANNELS = 13


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check required files and folders.
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found:\n{DATA_DIR}"
        )

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found:\n{CHECKPOINT_PATH}"
        )

    if not SPLITS_PATH.exists():
        raise FileNotFoundError(
            f"Dataset split file not found:\n{SPLITS_PATH}\n"
            "Use the same splits.csv that was used during training."
        )

    # Select CPU or GPU.
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
    print("Loading dataset...")

    # Load the same dataset splits used during training.
    _, _, test_loader, classes = get_loaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        seed=SEED,
        num_workers=0,
        num_channels=NUM_CHANNELS,
        splits_path=SPLITS_PATH,
    )

    if not classes:
        raise RuntimeError("No class names were loaded.")

    print(f"Number of classes: {len(classes)}")
    print(f"Test batches: {len(test_loader)}")

    # Create the model architecture.
    model = SmallCNN(
        num_classes=len(classes),
        in_channels=NUM_CHANNELS,
    ).to(device)

    # Load the saved model checkpoint.
    # weights_only=False is needed for checkpoints containing
    # metadata in addition to the model weights.
    try:
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        # Compatibility with older PyTorch versions.
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=device,
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Best model loaded successfully.")
    print("Evaluating test dataset...")

    all_predictions = []
    all_labels = []
    total_loss = 0.0
    total_samples = 0

    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(
            test_loader, start=1
        ):
            images = images.to(device)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            loss = criterion(outputs, labels)

            predictions = outputs.argmax(dim=1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_predictions.extend(
                predictions.cpu().numpy().tolist()
            )
            all_labels.extend(
                labels.cpu().numpy().tolist()
            )

            if (
                batch_index == 1
                or batch_index % 10 == 0
                or batch_index == len(test_loader)
            ):
                print(
                    f"Test batch {batch_index}/"
                    f"{len(test_loader)}"
                )

    if total_samples == 0:
        raise RuntimeError("The test dataset contains no samples.")

    y_true = np.asarray(all_labels)
    y_pred = np.asarray(all_predictions)

    test_loss = total_loss / total_samples
    test_accuracy = float(np.mean(y_true == y_pred))

    # Classification report


    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    report_path = OUTPUT_DIR / "classification_report.csv"

    with report_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            ["class", "precision", "recall", "f1-score", "support"]
        )

        for class_name in classes:
            row = report[class_name]
            writer.writerow(
                [
                    class_name,
                    row["precision"],
                    row["recall"],
                    row["f1-score"],
                    row["support"],
                ]
            )

        for average_name in ("macro avg", "weighted avg"):
            row = report[average_name]
            writer.writerow(
                [
                    average_name,
                    row["precision"],
                    row["recall"],
                    row["f1-score"],
                    row["support"],
                ]
            )

    # Confusion matrix

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
    )

    np.savetxt(
        OUTPUT_DIR / "confusion_matrix.csv",
        matrix,
        delimiter=",",
        fmt="%d",
    )

    fig, ax = plt.subplots(figsize=(12, 10))

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=classes,
    )

    display.plot(
        ax=ax,
        xticks_rotation=45,
        cmap="Blues",
        colorbar=True,
        values_format="d",
    )

    ax.set_title("EuroSAT MS - Test Confusion Matrix")
    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # Save summary

    summary = {
        "model": "SmallCNN",
        "dataset": str(DATA_DIR),
        "num_channels": NUM_CHANNELS,
        "num_classes": len(classes),
        "classes": classes,
        "best_epoch": checkpoint.get("epoch"),
        "test_samples": int(total_samples),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "test_accuracy_percent": float(test_accuracy * 100),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(
            report["weighted avg"]["f1-score"]
        ),
    }

    with (OUTPUT_DIR / "evaluation_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("\n" + "=" * 55)
    print("FINAL TEST RESULTS")
    print("=" * 55)
    print(f"Test samples : {total_samples}")
    print(f"Test loss    : {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Accuracy (%) : {test_accuracy * 100:.2f}%")
    print(f"Macro F1     : {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted F1  : {report['weighted avg']['f1-score']:.4f}")

    print("\nSaved files:")
    print(f"- {OUTPUT_DIR / 'evaluation_summary.json'}")
    print(f"- {report_path}")
    print(f"- {OUTPUT_DIR / 'confusion_matrix.csv'}")
    print(f"- {OUTPUT_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
