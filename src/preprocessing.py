"""
Milestone 1 - Preprocessing (Hazem)

Pipeline per image:
    1) load            (.tif multispectral via rasterio, or .jpg/.png via PIL)
    2) atmospheric correction (optional, simple Dark Object Subtraction)
    3) resize to one consistent size
    4) normalize
    5) save as .npy (float32) keeping the class-folder structure

Example:
    python src/preprocessing.py --input data/raw --output data/processed --size 64 --correction none --norm scale

Needs: numpy, pillow, opencv-python, rasterio (rasterio only for .tif files)
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

try:
    import rasterio
except ImportError:  # only needed for .tif files
    rasterio = None

EXTS = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}


def load_image(path: Path):
    """Return (image as H x W x C float32 array, max value used for scaling)."""
    if path.suffix.lower() in {".tif", ".tiff"}:
        if rasterio is None:
            raise ImportError("Install rasterio first: pip install rasterio")
        with rasterio.open(path) as src:
            arr = src.read().astype(np.float32)  # shape (C, H, W)
        return np.transpose(arr, (1, 2, 0)), 10000.0  # Sentinel-2 reflectance scale
    img = np.array(Image.open(path).convert("RGB"), dtype=np.float32)
    return img, 255.0  # 8-bit images


def dark_object_subtraction(img: np.ndarray, percentile: float = 1.0):
    """Simple atmospheric correction: subtract each band's darkest values (haze)."""
    dark = np.percentile(img.reshape(-1, img.shape[-1]), percentile, axis=0)
    return np.clip(img - dark, 0, None)


def resize_image(img: np.ndarray, size: int):
    """Resize every band to size x size."""
    h, w = img.shape[:2]
    interp = cv2.INTER_AREA if size < min(h, w) else cv2.INTER_LINEAR
    bands = [
        cv2.resize(img[:, :, b], (size, size), interpolation=interp)
        for b in range(img.shape[2])
    ]
    return np.stack(bands, axis=-1)


def normalize(img: np.ndarray, scale: float, method: str):
    if method == "scale":  # divide by 10000 (or 255) -> values 0..1
        return np.clip(img / scale, 0.0, 1.0)
    # "minmax": stretch each band of this image to 0..1
    mn = img.min(axis=(0, 1), keepdims=True)
    mx = img.max(axis=(0, 1), keepdims=True)
    return (img - mn) / (mx - mn + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--correction", choices=["none", "dos"], default="none")
    ap.add_argument("--norm", choices=["scale", "minmax"], default="scale")
    args = ap.parse_args()

    in_dir, out_dir = Path(args.input), Path(args.output)
    files = [p for p in in_dir.rglob("*") if p.suffix.lower() in EXTS]
    if not files:
        print(f"No images found in {in_dir}")
        return

    for i, p in enumerate(files, 1):
        img, scale = load_image(p)
        if args.correction == "dos":
            img = dark_object_subtraction(img)
        img = resize_image(img, args.size)
        img = normalize(img, scale, args.norm).astype(np.float32)

        out_path = out_dir / p.relative_to(in_dir).with_suffix(".npy")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, img)
        if i % 500 == 0:
            print(f"{i}/{len(files)} done")

    print(f"Finished: {len(files)} images -> {out_dir}")


if __name__ == "__main__":
    main()
