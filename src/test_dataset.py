from pathlib import Path
import sys
import torch

PROJECT_ROOT = Path(
    r"C:\Users\HP\Downloads\satellite_empty"
)

sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import get_loaders

DATA_DIR = (
    PROJECT_ROOT / "EuroSAT_MS" / "EuroSAT_MS"
)

train_loader, val_loader, test_loader, classes = get_loaders(
    data_dir=DATA_DIR,
    batch_size=8,
    seed=42,
    num_workers=0,
    num_channels=13,
    splits_path=PROJECT_ROOT / "data" / "processed" / "splits.csv",
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