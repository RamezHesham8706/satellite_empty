from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import get_loaders, resolve_data_dir


def main():
    splits_path = PROJECT_ROOT / "data" / "processed" / "splits.csv"
    train_loader, val_loader, test_loader, classes = get_loaders(
        data_dir=resolve_data_dir(),
        batch_size=8,
        seed=42,
        num_workers=0,
        num_channels=13,
        splits_path=splits_path,
    )

    images, labels = next(iter(train_loader))

    print("Classes:", classes)
    print("Number of classes:", len(classes))
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)
    print("Image dtype:", images.dtype)
    print("Image min:", images.min().item())
    print("Image max:", images.max().item())
    print("Training batches:", len(train_loader))
    print("Validation batches:", len(val_loader))
    print("Testing batches:", len(test_loader))


if __name__ == "__main__":
    main()
