from pathlib import Path
import csv

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

try:
    import rasterio
except ImportError:
    rasterio = None


# Paths and configuration

ROOT = Path(__file__).resolve().parents[1]

SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".npy", ".npz"}

DEFAULT_DATA_DIRS = [
    ROOT / "data" / "raw" / "EuroSAT_MS",
    ROOT / "data" / "raw" / "EuroSAT" / "EuroSAT_MS",
    ROOT / "data" / "raw" / "eurosat_ms",
    ROOT / "data" / "processed",
]

CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]


# Resolve dataset directory

def resolve_data_dir(data_dir=None):
    """
    Resolve the dataset directory.

    If data_dir is provided, use it.
    Otherwise, search known EuroSAT MS locations.
    """

    if data_dir is not None:
        path = Path(data_dir).expanduser().resolve()

        if not path.is_dir():
            raise FileNotFoundError(
                f"Dataset directory does not exist: {path}"
            )

        return path

    for path in DEFAULT_DATA_DIRS:
        if path.is_dir():
            # Select a directory only if it contains supported images.
            if any(
                p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
                for p in path.rglob("*")
            ):
                return path

    raise FileNotFoundError(
        "Could not find EuroSAT MS dataset.\n"
        "Pass the actual dataset path to get_loaders(data_dir=...)."
    )


# Read multispectral images

def load_multispectral_image(path):
    """
    Load one image as a float32 NumPy array.

    Supported formats:
        TIFF: read using rasterio.
        NPY: load a saved NumPy array.
        NPZ: load the first array stored in the archive.

    Output:
        Array with shape (Channels, Height, Width).
    """

    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".tif", ".tiff"}:
        if rasterio is None:
            raise ImportError(
                "TIFF files require rasterio. "
                "Install it using: pip install rasterio"
            )

        with rasterio.open(path) as src:
            image = src.read().astype(np.float32)

    elif suffix == ".npy":
        image = np.load(path, allow_pickle=False).astype(np.float32)

    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if not archive.files:
                raise ValueError(f"Empty NPZ archive: {path}")

            # Prefer an array named 'image' if present.
            key = "image" if "image" in archive.files else archive.files[0]
            image = archive[key].astype(np.float32)

    else:
        raise ValueError(f"Unsupported image format: {suffix}")

    if image.ndim != 3:
        raise ValueError(
            f"Expected a 3D multispectral image, got shape {image.shape} "
            f"from {path}"
        )

    # Accept either CHW or HWC layout.
    if image.shape[0] in (3, 13):
        pass
    elif image.shape[-1] in (3, 13):
        image = np.transpose(image, (2, 0, 1))
    else:
        raise ValueError(
            f"Cannot determine channel dimension for {path}: "
            f"shape={image.shape}"
        )

    if not np.isfinite(image).all():
        raise ValueError(f"Image contains NaN or infinite values: {path}")

    return image


# Collect image paths and labels

def discover_samples(data_dir):
    """
    Expected directory structure:

    EuroSAT_MS/
        AnnualCrop/
            image1.tif
        Forest/
            image2.tif
        ...
        SeaLake/
            image3.tif

    The class directory name becomes the label.
    """

    data_dir = Path(data_dir)
    samples = []

    class_dirs = sorted(
        [
            p for p in data_dir.iterdir()
            if p.is_dir() and p.name in CLASS_NAMES
        ],
        key=lambda p: p.name.lower()
    )

    if not class_dirs:
        raise ValueError(
            f"No class folders found in {data_dir}.\n"
            "Expected folders such as AnnualCrop, Forest, and SeaLake."
        )

    # Keep one consistent class-to-index mapping.
    found_classes = [name for name in CLASS_NAMES if name in {
        p.name for p in class_dirs
    }]

    class_to_idx = {
        class_name: index
        for index, class_name in enumerate(found_classes)
    }

    for class_name in found_classes:
        class_dir = data_dir / class_name

        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                samples.append(
                    (path, class_to_idx[class_name], class_name)
                )

    if not samples:
        raise ValueError(f"No supported images found in {data_dir}")

    return samples, found_classes


# Dataset class

class EuroSATMSDataset(Dataset):
    def __init__(
        self,
        samples,
        indices,
        num_channels=13,
        augment=False,
    ):
        self.samples = samples
        self.indices = list(indices)
        self.num_channels = num_channels
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        sample_index = self.indices[index]

        path, label, _ = self.samples[sample_index]

        image = load_multispectral_image(path)

        if image.shape[0] != self.num_channels:
            raise ValueError(
                f"Expected {self.num_channels} channels, "
                f"but got {image.shape[0]} in {path}"
            )

        # Scale raw Sentinel-2 reflectance if necessary.
        # Preprocessed arrays in [0, 1] remain unchanged.
        if image.min() < 0:
            raise ValueError(f"Unexpected negative pixel values in {path}")

        if image.max() > 1.0:
            image = image / 10000.0

        image = np.clip(image, 0.0, 1.0)

        image = torch.from_numpy(
            np.ascontiguousarray(image, dtype=np.float32)
        )

        # Spatial augmentation only: apply the same operation
        # to all spectral bands.
        if self.augment:
            if torch.rand(()) < 0.5:
                image = torch.flip(image, dims=[2])

            if torch.rand(()) < 0.5:
                image = torch.flip(image, dims=[1])

        return image, torch.tensor(label, dtype=torch.long)


# Save data splits

def save_splits(samples, train_idx, val_idx, test_idx, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    split_names = {}

    for name, indices in [
        ("train", train_idx),
        ("val", val_idx),
        ("test", test_idx),
    ]:
        for index in indices:
            split_names[index] = name

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["path", "class_name", "label", "split"])

        for index, (path, label, class_name) in enumerate(samples):
            writer.writerow([
                str(path.resolve()),
                class_name,
                label,
                split_names[index],
            ])


# Main loader function

def get_loaders(
    data_dir=None,
    batch_size=32,
    seed=42,
    num_workers=0,
    num_channels=13,
    splits_path=None,
):
    """
    Return:
        train_loader, val_loader, test_loader, classes
    """

    data_dir = resolve_data_dir(data_dir)
    samples, classes = discover_samples(data_dir)

    labels = [label for _, label, _ in samples]
    indices = np.arange(len(samples))

    train_idx, rest_idx = train_test_split(
        indices,
        test_size=0.30,
        random_state=seed,
        stratify=labels,
    )

    rest_labels = [labels[i] for i in rest_idx]

    val_idx, test_idx = train_test_split(
        rest_idx,
        test_size=0.50,
        random_state=seed,
        stratify=rest_labels,
    )

    if splits_path is not None:
        save_splits(samples, train_idx, val_idx, test_idx, splits_path)

    train_dataset = EuroSATMSDataset(
        samples,
        train_idx,
        num_channels=num_channels,
        augment=True,
    )

    val_dataset = EuroSATMSDataset(
        samples,
        val_idx,
        num_channels=num_channels,
        augment=False,
    )

    test_dataset = EuroSATMSDataset(
        samples,
        test_idx,
        num_channels=num_channels,
        augment=False,
    )

    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    return train_loader, val_loader, test_loader, classes
