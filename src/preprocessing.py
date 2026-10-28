# preprocessing for EuroSAT MS (13 band tif images)
# run it from the project folder:  python src/preprocessing.py

from pathlib import Path

import cv2
import numpy as np
import rasterio

INPUT_FOLDER = "data/raw/EuroSAT_MS"
OUTPUT_FOLDER = "data/processed"
IMAGE_SIZE = 64
REMOVE_HAZE = True      # set to False to skip the atmospheric correction
SCALE = 10000           # sentinel-2 values are reflectance * 10000


def load_image(path):
    with rasterio.open(path) as f:
        img = f.read().astype(np.float32)    # comes as (13, 64, 64)
    return np.transpose(img, (1, 2, 0))      # we want (64, 64, 13)


def find_haze(files):
    # atmospheric correction (dark object subtraction):
    # the darkest pixels should be close to 0, whatever is above that is haze.
    # taking every 50th image is enough to measure it
    pixels = []
    for path in files[::50]:
        img = load_image(path)
        pixels.append(img.reshape(-1, img.shape[2]))
    pixels = np.concatenate(pixels)
    return np.percentile(pixels, 1, axis=0)   # one number per band


def remove_haze(img, haze):
    return np.maximum(img - haze, 0)


def resize(img, size):
    if img.shape[0] == size and img.shape[1] == size:
        return img    # already the right size

    if size < img.shape[0]:
        method = cv2.INTER_AREA
    else:
        method = cv2.INTER_LINEAR

    bands = []
    for b in range(img.shape[2]):
        bands.append(cv2.resize(img[:, :, b], (size, size), interpolation=method))
    return np.stack(bands, axis=2)


def normalize(img):
    return np.clip(img / SCALE, 0, 1)


def preprocess(img, haze=None):
    if haze is not None:
        img = remove_haze(img, haze)
    img = resize(img, IMAGE_SIZE)
    img = normalize(img)
    return img.astype(np.float32)


def main():
    in_dir = Path(INPUT_FOLDER)
    out_dir = Path(OUTPUT_FOLDER)

    files = sorted(in_dir.rglob("*.tif"))
    if not files:
        print("no tif images found in", in_dir)
        return
    print(len(files), "images found")

    haze = None
    if REMOVE_HAZE:
        haze = find_haze(files)
        print("haze per band:", np.round(haze))

    for i, path in enumerate(files, start=1):
        result = preprocess(load_image(path), haze)

        # keep the same class folder, just save it as .npy
        save_path = out_dir / path.relative_to(in_dir).with_suffix(".npy")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, result)

        if i % 500 == 0:
            print(i, "/", len(files))

    print("done,", len(files), "images saved in", out_dir, "shape:", result.shape)


if __name__ == "__main__":
    main()